'use client';
import { ReloadIcon } from '@radix-ui/react-icons';
import { debounce } from 'lodash';
import React, {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from 'react';
import { z } from 'zod';

import { Message } from '@/components/chat/chat';
import { Footer } from '@/components/main-stepper/main-stepper-footer';
import StepRenderer from '@/components/main-stepper/step-renderer';
import { Button } from '@/components/ui/button';
import { Step, Stepper } from '@/components/ui/stepper';
import { useStepperForm } from '@/contexts/stepper-form-context';
import {
  StructuredInputFormProvider,
  StructuredInputFormSchema,
} from '@/contexts/structured-input-form-context';
import { useUser } from '@/contexts/user-context';
import {
  CutTranscriptLinearWorkflowStepOutput,
  CutTranscriptLinearWorkflowStreamingOutput,
  ExportableStepWrapper,
  ExportResults,
  FrontendStepOutput,
  FrontendWorkflowProjection,
  FrontendWorkflowState,
} from '@/gen/openapi/api';
import {
  getLatestState,
  getWorkflowDetails,
  resetWorkflow,
  revertStepInBackend,
  step,
} from '@/lib/api';
import { decodeStreamAsJSON } from '@/lib/streams';
import { RevertStepParams, StepData } from '@/lib/types';

function stepIndexFromState(state: FrontendWorkflowState): number {
  if (!state.all_steps) {
    throw new Error('state does not contain all_steps');
  }
  if (state && state.outputs) {
    return state.outputs.length - 1;
  }
  return -1;
}

function stepIndexFromName(
  stepName: string,
  allSteps: Array<ExportableStepWrapper>
): number {
  const currentStepIndex = allSteps.findIndex((step) => step.name === stepName);
  if (currentStepIndex === -1) {
    console.error(`Could not find step ${stepName} in steps array`, allSteps);
    return -1;
  }
  return currentStepIndex;
}

function stepOutputFromIndex(
  stepIndex: number,
  state: FrontendWorkflowState
): CutTranscriptLinearWorkflowStepOutput {
  if (!state.all_steps) {
    throw new Error('state does not contain all_steps');
  }
  if (!state.outputs) {
    throw new Error('state does not contain outputs');
  }
  if (stepIndex > state.outputs.length || stepIndex < 0) {
    throw new Error('stepIndex out of bounds');
  }
  return state.outputs[stepIndex];
}

function mappedExportResultFromIndex(
  stepIndex: number,
  state: FrontendWorkflowState
): Record<string, any> {
  if (!state.all_steps) {
    throw new Error('state does not contain all_steps');
  }
  if (!state.mapped_export_result) {
    throw new Error('state does not contain mapped_export_result');
  }
  if (stepIndex > state.mapped_export_result.length || stepIndex < 0) {
    throw new Error('stepIndex out of bounds');
  }
  return state.mapped_export_result[stepIndex];
}

/**
 * Main stepper component.
 * - Get all steps and maintain their state
 * - Render the current step
 * - Handle stepping through steps
 * - Handle retrying / undoing
 */
export default function MainStepper({
  initialProjectId,
  initialProjectName,
  workflowId,
}: {
  initialProjectId: string;
  workflowId: string;
  initialProjectName: string;
}) {
  const { userData } = useUser();
  // const [workflowId, setWorkflowId] = useState<string>(initialWorkflowId);
  // const [projectName, setProjectName] = useState<string>(initialProjectName);
  // const [projectId, setProjectId] = useState<string>(initialProjectId);
  const [workflow, setWorkflow] = useState<FrontendWorkflowProjection | null>(
    null
  );
  const { stepperFormValues } = useStepperForm();
  const [latestState, setLatestState] = useState<FrontendWorkflowState | null>(
    null
  );
  const [stepInputPrompt, setStepInputPrompt] = useState<string>('');
  const [trueStepIndex, setTrueStepIndex] = useState<number>(-1);
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(-1);
  const [userMessage, setUserMessage] = useState<string>('');

  const [stepOutput, setStepOutput] = useState<FrontendStepOutput | null>(null);
  const [mappedExportResult, setMappedExportResult] = useState<Record<
    string,
    any
  > | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [backendMessage, setBackendMessage] = useState<string>('');
  const [hasCompletedAllSteps, setHasCompletedAllSteps] =
    useState<boolean>(false);
  const [structuredInputFormData, setStructuredInputFormData] =
    useState<z.infer<typeof StructuredInputFormSchema>>();

  const handleStructuredInputFormValueChange = debounce(
    (values: z.infer<typeof StructuredInputFormSchema>) => {
      setStructuredInputFormData(values);
    },
    300
  );

  const [chatMessages, setChatMessages] = useState<Message[]>([]);

  const setAIMessageCallback = (aiMessage: string) => {
    setChatMessages((prevMessages) => [
      ...prevMessages,
      { sender: 'AI', text: aiMessage },
    ]);
  };

  const userParams = useMemo(
    () => ({
      user_email: userData.email,
      workflow_id: workflow?.id || '',
      project_id: workflow?.project_id,
      project_name: workflow?.project_name,
      video_hash: workflow?.video_hash || null,
    }),
    [userData.email, workflow]
  );
  const fetchedInitialState = useRef(false);
  const allowRunningFromCurrentStepIndexChange = useRef(false);
  const prevExportResult = useRef<ExportResults>();
  const prevExportCallId = useRef<string>();

  useEffect(() => {
    async function fetchAndSetWorkflow() {
      const workflow = await getWorkflowDetails(workflowId);
      setWorkflow(workflow);
    }
    if (workflowId) {
      fetchAndSetWorkflow();
    }
  }, [workflowId]);
  useEffect(() => {
    async function fetchLatestStateAndMaybeSetCurrentStepIndexAndStepOutput() {
      // since we include currentStepIndex as a dependency
      // but set its value in the first successful call to this effect hook,
      // the 2nd call to this hook should be a noop
      if (
        fetchedInitialState.current &&
        !allowRunningFromCurrentStepIndexChange.current
      ) {
        allowRunningFromCurrentStepIndexChange.current = true;
        return;
      }
      const data = await getLatestState(userParams.workflow_id);
      if (!data || Object.keys(data).length === 0) return;
      if (!fetchedInitialState.current) {
        let stepIndex = -1;
        try {
          stepIndex = stepIndexFromState(data);
        } catch (error) {
          console.log(error);
        }

        setCurrentStepIndex(stepIndex);

        if (data.outputs && data.outputs.length) {
          setStepOutput(data.outputs[data.outputs.length - 1]);
        }
        if (data.mapped_export_result && data.mapped_export_result.length) {
          setMappedExportResult(
            data.mapped_export_result[data.mapped_export_result.length - 1]
          );
        }
      }
      setLatestState(data);
      fetchedInitialState.current = true;
    }
    fetchLatestStateAndMaybeSetCurrentStepIndexAndStepOutput();
  }, [userData, userParams, workflow?.video_hash, currentStepIndex]);

  const prevTrueStepIndex = useRef<number>(trueStepIndex);

  const handleStepStream = useCallback(
    async (reader: ReadableStreamDefaultReader) => {
      activePromptDispatch({ type: 'restart', value: '' });
      const finalState: FrontendWorkflowState | null = await decodeStreamAsJSON(
        reader,
        (value: CutTranscriptLinearWorkflowStreamingOutput) => {
          if (value?.partial_step_output) {
            // we can log this maybe
          } else if (value?.partial_backend_output) {
            let output = value.partial_backend_output;
            const currentSubstep = output.current_substep;
            if (currentSubstep) {
              const newIndex = stepIndexFromName(
                currentSubstep.step_name,
                latestState?.all_steps || []
              );
              setCurrentStepIndex(newIndex);
            }
            if (output.chunk === null || output.chunk == 0) {
              setBackendMessage(output.value || '');
            }
          } else if (value?.partial_llm_output) {
            let output = value.partial_llm_output;
            if (output.chunk === null || output.chunk == 0) {
              activePromptDispatch({ type: 'append', value: output.value });
              //  TODO: for some reason this is not triggering StepperForm+systemPrompt to reload
            }
          }
        }
      );
      setLatestState(finalState);
      setStepOutput(
        finalState?.outputs?.length
          ? finalState.outputs[finalState.outputs.length - 1]
          : null
      );
      setMappedExportResult(
        finalState?.mapped_export_result?.length
          ? finalState.mapped_export_result[
              finalState.mapped_export_result.length - 1
            ]
          : null
      );

      setIsLoading(false);
      return finalState;
    },
    [latestState?.all_steps]
  );

  const advanceStep = useCallback(
    async (stepIndex: number) => {
      setIsLoading(true);
      setCurrentStepIndex(stepIndex);
      setStepOutput(null);
      setMappedExportResult(null);
      const stepData: StepData = {
        user_input:
          stepperFormValues.feedback !== undefined &&
          stepperFormValues.feedback !== null
            ? stepperFormValues.feedback
            : '',
        streaming: true,
        ignore_running_workflows: true,
        retry_step: false,
        advance_until: stepIndex,
      };
      try {
        await step(userParams.workflow_id, stepData, handleStepStream);
      } catch (error) {
        console.error('error in step', error);
      }
    },
    [userParams, stepperFormValues, handleStepStream]
  );

  useEffect(() => {
    async function setPageStateFromBackendState() {
      if (!latestState || !latestState.all_steps) return;

      const stepIndex = stepIndexFromState(latestState);
      if (prevTrueStepIndex.current != stepIndex) {
        setTrueStepIndex(stepIndex);
        prevTrueStepIndex.current = stepIndex;
      }

      if (stepIndex === latestState.all_steps.length - 1) {
        setHasCompletedAllSteps(true);
      }
      let currentStep =
        currentStepIndex >= 0 ? latestState.all_steps[currentStepIndex] : null;
      setStepInputPrompt(currentStep?.input_prompt || '');

      if (isLoading) {
        return;
      }
      if (!latestState.outputs || !latestState.outputs.length) {
        await advanceStep(0);
        return;
      }
      const lastOutput = latestState.outputs[latestState.outputs.length - 1];
      if (
        lastOutput.export_result &&
        lastOutput.export_result != prevExportResult.current
      ) {
        prevExportResult.current = lastOutput.export_result;
      } else if (
        lastOutput.export_call_id &&
        lastOutput.export_call_id != prevExportCallId.current
      ) {
        prevExportCallId.current = lastOutput.export_call_id;
      }
    }
    setPageStateFromBackendState();
  }, [latestState, currentStepIndex, isLoading, advanceStep]);

  useEffect(() => {
    if (stepOutput === null) {
      setChatMessages([]);
      return;
    }
    const newMessages: Message[] = stepInputPrompt
      ? [{ sender: 'AI', text: stepInputPrompt }]
      : [];
    if (stepOutput?.full_conversation) {
      stepOutput.full_conversation.forEach((msg) => {
        newMessages.push({ sender: msg.role, text: msg.value });
      });
    }
    setChatMessages(newMessages);
  }, [stepOutput, stepInputPrompt]);

  type ActivePromptAction =
    | { type: 'append'; value: string }
    | { type: 'restart'; value: string };

  // TODO change activePrompt name to outputText
  const [activePrompt, activePromptDispatch] = useReducer(
    (state: string, action: ActivePromptAction) => {
      switch (action.type) {
        case 'append':
          return state + action.value;
        case 'restart':
          return action.value;
        default:
          throw new Error('Unhandled action type');
      }
    },
    ''
  );

  // TODO combine this method with the form once we have structured input
  async function advanceOrRetryStep(options: { useStructuredInput: boolean }) {
    // if (options.stepIndex === null || options.stepIndex === undefined) {
    // throw new Error('must provide stepIndex to advanceOrRetryStep');
    // }

    // TODO: stepperFormValues.feedback is cut off- doesn't include last character
    setIsLoading(true);
    let retry = false;
    if (trueStepIndex >= currentStepIndex) {
      retry = true;
    }
    console.log(
      'advanceOrRetryStep retry',
      retry,
      'trueStepIndex',
      trueStepIndex,
      'advance_until',
      currentStepIndex,
      'useStructuredUserInput',
      options.useStructuredInput,
      'useStructuredUserInput',
      structuredInputFormData
    );

    //setCurrentStepIndex(options.stepIndex);
    setStepOutput(null);
    setMappedExportResult(null);
    const stepData: StepData = {
      user_input: userMessage || '',
      structured_user_input: options.useStructuredInput
        ? structuredInputFormData
        : undefined,
      streaming: true,
      ignore_running_workflows: true,
      retry_step: retry,
      advance_until: currentStepIndex,
    };
    console.log('stepData', stepData);
    try {
      await step(userParams.workflow_id, stepData, async (reader) => {
        const finalState = await handleStepStream(reader);
        const stepOutput = finalState?.outputs?.length
          ? finalState.outputs[finalState.outputs.length - 1]
          : null;
        const aiMessage = stepOutput?.full_conversation?.length
          ? stepOutput.full_conversation[
              stepOutput.full_conversation.length - 1
            ].value
          : '';
        let currentStepIndex = -1;
        if (finalState) {
          currentStepIndex = stepIndexFromState(finalState);
        }
        setCurrentStepIndex(currentStepIndex);
        setAIMessageCallback(aiMessage);
      });
    } catch (error) {
      console.error('error in step', error);
    }
  }

  async function restart() {
    setIsLoading(true);
    activePromptDispatch({ type: 'restart', value: '' });
    setStepOutput(null);
    setMappedExportResult(null);
    await resetWorkflow(userParams.workflow_id);
    const newState = await getLatestState(userParams.workflow_id);
    setLatestState(newState);
    setCurrentStepIndex(-1);
    setIsLoading(false);
  }

  async function revertStep(toBeforeRetries: boolean) {
    setIsLoading(true);
    activePromptDispatch({ type: 'restart', value: '' });
    setStepOutput(null);
    setMappedExportResult(null);
    await revertStepInBackend({
      to_before_retries: toBeforeRetries,
      workflow_id: userParams.workflow_id,
    } as RevertStepParams);
    const latestState = await getLatestState(userParams.workflow_id);
    setLatestState(latestState);
    setCurrentStepIndex(stepIndexFromState(latestState));
    setIsLoading(false);
  }

  async function undoLastStep() {
    await revertStep(false);
  }

  function updateStepOutput(stepIndex: number, state: FrontendWorkflowState) {
    let stepOutput: FrontendStepOutput | null = null;
    try {
      stepOutput = stepOutputFromIndex(stepIndex, state);
    } catch (error) {
      console.error(error);
    }
    setStepOutput(stepOutput);
    let mappedExportResult: Record<string, any> | null = null;
    try {
      mappedExportResult = mappedExportResultFromIndex(stepIndex, state);
    } catch (error) {
      console.error(error);
    }
    setMappedExportResult(mappedExportResult);
  }

  async function onPrevStep() {
    if (currentStepIndex === -1 || isLoading || !latestState?.all_steps) {
      return;
    }
    setCurrentStepIndex(currentStepIndex - 1);
    activePromptDispatch({ type: 'restart', value: '' });
    updateStepOutput(currentStepIndex - 1, latestState);
  }
  async function onNextStep() {
    if (
      isLoading ||
      !latestState?.all_steps ||
      currentStepIndex == trueStepIndex + 1
    ) {
      return;
    }
    if (currentStepIndex > trueStepIndex + 1) {
      setCurrentStepIndex(trueStepIndex + 1);
      return;
    }
    setCurrentStepIndex(currentStepIndex + 1);
    // TODO: When the backend is modified to return conversations per-step,
    // we should update the conversation messages here
    // (currently activePromptDispatch but kasp is adding better conversation UX)
    // using the messages in latestState for this particular step
    activePromptDispatch({ type: 'restart', value: '' });
    if (currentStepIndex + 1 <= trueStepIndex) {
      updateStepOutput(currentStepIndex + 1, latestState);
    } else {
      setStepOutput(null);
      setMappedExportResult(null);
    }
  }

  const workflowInitialized = workflow && latestState?.all_steps !== undefined;
  function onCancelStep() {
    throw new Error('not implemented');
  }

  return (
    <div className="flex w-full flex-col gap-4">
      <div className="flex gap-3 w-full justify-between mb-3 items-center">
        Video: {workflow?.video_filename}
        {workflowInitialized ? (
          <div className="flex gap-3 items-center">
            <Button variant="outline" onClick={restart} disabled={isLoading}>
              <ReloadIcon className="mr-2" />
              Restart
            </Button>
          </div>
        ) : null}
      </div>

      {workflowInitialized ? (
        <StructuredInputFormProvider
          onFormDataChange={handleStructuredInputFormValueChange}
          stepOutput={stepOutput}
          mappedExportResult={mappedExportResult}
          userParams={userParams}
        >
          <Stepper
            initialStep={currentStepIndex > -1 ? currentStepIndex : 0}
            steps={latestState.all_steps.map((step: ExportableStepWrapper) => {
              return { label: step.human_readable_name };
            })}
            orientation="vertical"
          >
            {latestState.all_steps.map(
              (step: ExportableStepWrapper, index: number) => (
                <Step key={step.name} label={step.human_readable_name}>
                  <div className="grid w-full gap-2">
                    <StepRenderer
                      step={step}
                      stepIndex={index}
                      stepInputPrompt={stepInputPrompt}
                      outputText={activePrompt} // TODO change activePrompt name to outputText
                      stepOutput={stepOutput}
                      chatMessages={chatMessages}
                      onSubmit={advanceOrRetryStep}
                      isNewStep={trueStepIndex < index}
                      isLoading={isLoading}
                      backendMessage={backendMessage}
                      isInitialized={workflowInitialized}
                      onCancelStep={onCancelStep}
                      setUserMessage={setUserMessage}
                      userMessage={userMessage}
                      footer={
                        <Footer
                          userMessage={userMessage}
                          currentStepIndex={currentStepIndex}
                          trueStepIndex={trueStepIndex}
                          onPrevStep={onPrevStep}
                          onNextStep={onNextStep}
                          undoLastStep={undoLastStep}
                          hasCompletedAllSteps={hasCompletedAllSteps}
                          totalNSteps={latestState.all_steps!.length}
                          userParams={userParams}
                          stepName={step.name}
                          isLoading={isLoading}
                          onSubmit={advanceOrRetryStep}
                        />
                      }
                    />
                  </div>
                </Step>
              )
            )}
          </Stepper>
        </StructuredInputFormProvider>
      ) : null}
    </div>
  );
}
