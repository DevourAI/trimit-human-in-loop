'use client';
import { DownloadIcon, ReloadIcon } from '@radix-ui/react-icons';
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
import { FormSchema } from '@/components/ui/stepper-form';
import { useStepperForm } from '@/contexts/stepper-form-context';
import {
  StructuredInputFormProvider,
  StructuredInputFormSchema,
} from '@/contexts/structured-input-form-context';
import { useUser } from '@/contexts/user-context';
import {
  CutTranscriptLinearWorkflowStepOutput,
  CutTranscriptLinearWorkflowStepOutputExportResultValue,
  CutTranscriptLinearWorkflowStreamingOutput,
  ExportableStepWrapper,
  FrontendStepOutput,
  FrontendWorkflowProjection,
  FrontendWorkflowState,
} from '@/gen/openapi/api';
import {
  getLatestState,
  getWorkflowDetails,
  resetWorkflow,
  revertStepInBackend,
  revertStepToInBackend,
  step,
} from '@/lib/api';
import { decodeStreamAsJSON } from '@/lib/streams';
import { RevertStepParams, RevertStepToParams } from '@/lib/types';

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

function stepNameFromIndex(
  allSteps: Array<ExportableStepWrapper>,
  stepIndex: number
): string {
  return allSteps[stepIndex].name;
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

/**
 * Main stepper component.
 * - Get all steps and maintain their state
 * - Render the current step
 * - Handle stepping through steps
 * - Handle retrying / undoing
 */
export default function MainStepper({ projectId }: { projectId: string }) {
  const { userData } = useUser();
  const [project, setProject] = useState<FrontendWorkflowProjection | null>(
    null
  );
  const { stepperFormValues } = useStepperForm();
  const [latestState, setLatestState] = useState<FrontendWorkflowState | null>(
    null
  );
  const [userFeedbackRequest, setUserFeedbackRequest] = useState<string>('');
  const [stepInputPrompt, setStepInputPrompt] = useState<string>('');
  const [trueStepIndex, setTrueStepIndex] = useState<number>(-1);
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(-1);
  const [userMessage, setUserMessage] = useState<string>('');

  const [latestExportResult, setLatestExportResult] = useState<
    Record<string, any>
  >({});
  const [latestExportCallId, setLatestExportCallId] = useState<string>('');
  const [stepOutput, setStepOutput] = useState<FrontendStepOutput | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [backendMessage, setBackendMessage] = useState<string>('');
  const [currentStepFormValues, setCurrentStepFormValues] = useState<
    z.infer<typeof FormSchema>
  >({});
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

  const chatInitialMessages = stepInputPrompt
    ? [{ sender: 'AI', text: stepInputPrompt }]
    : [];
  if (stepOutput?.full_conversation) {
    stepOutput.full_conversation.forEach((msg) => {
      chatInitialMessages.push({ sender: msg.role, text: msg.value });
    });
  }

  const [chatMessages, setChatMessages] =
    useState<Message[]>(chatInitialMessages);

  const setAIMessageCallback = (aiMessage: string) => {
    setChatMessages((prevMessages) => [
      ...prevMessages,
      { sender: 'AI', text: aiMessage },
    ]);
  };

  const userParams = useMemo(
    () => ({
      user_email: userData.email,
      workflow_id: project?.id || null,
      video_hash: project?.video_hash || null,
    }),
    [userData.email, project]
  );
  const fetchedInitialState = useRef(false);
  const allowRunningFromCurrentStepIndexChange = useRef(false);
  const prevExportResult = useRef<{
    [key: string]: CutTranscriptLinearWorkflowStepOutputExportResultValue;
  }>();
  const prevExportCallId = useRef<string>();

  useEffect(() => {
    async function fetchAndSetProject() {
      const project = await getWorkflowDetails(projectId);
      setProject(project);
    }
    if (projectId) {
      fetchAndSetProject(projectId);
    }
  }, [projectId]);
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
      }
      setLatestState(data);
      fetchedInitialState.current = true;
    }
    fetchLatestStateAndMaybeSetCurrentStepIndexAndStepOutput();
  }, [userData, userParams, project?.video_hash, currentStepIndex]);

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
                latestState.all_steps
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
        setLatestExportResult(lastOutput.export_result);

        prevExportResult.current = lastOutput.export_result;
      } else if (
        lastOutput.export_call_id &&
        lastOutput.export_call_id != prevExportCallId.current
      ) {
        setLatestExportCallId(lastOutput.export_call_id);
        prevExportCallId.current = lastOutput.export_call_id;
      }
    }
    setPageStateFromBackendState();
  }, [latestState, currentStepIndex, isLoading, advanceStep]);

  useEffect(() => {
    setUserFeedbackRequest(stepOutput?.user_feedback_request || '');
  }, [stepOutput]);

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

  const revertStepTo = useCallback(
    async (stepIndex: number) => {
      setIsLoading(true);
      if (!latestState || !latestState.all_steps) {
        throw new Error(
          "can't revert unless latestState.all_steps is available"
        );
      }
      const stepName = stepNameFromIndex(latestState.all_steps, stepIndex);
      const success = await revertStepToInBackend({
        step_name: stepName,
        ...userParams,
      } as RevertStepToParams);
      if (success) {
        activePromptDispatch({ type: 'restart', value: '' });
        const latestState = await getLatestState(userParams.workflow_id);
        setLatestState(latestState);
        setCurrentStepIndex(stepIndexFromState(latestState));
      }
      setIsLoading(false);
      return success;
    },
    [latestState, userParams]
  );

  async function retryStep(data: z.infer<typeof FormSchema>) {
    // TODO: stepperFormValues.feedback is cut off- doesn't include last character
    setIsLoading(true);
    if (trueStepIndex > currentStepIndex) {
      const success = await revertStepTo(currentStepIndex + 1);
      if (!success) {
        setIsLoading(false);
        return;
      }
    }
    const stepData: StepData = {
      user_input:
        data.feedback !== undefined && data.feedback !== null
          ? data.feedback
          : '',
      streaming: true,
      force_restart: false,
      ignore_running_workflows: true,
      retry_step: true,
    };
    try {
      await step(userParams, stepData, handleStepStream);
    } catch (error) {
      console.error('error in step', error);
    }
  }
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
        console.log('finalState', finalState);
        console.log('stepIndexFromFinalState', stepIndexFromState(finalState));
        setCurrentStepIndex(stepIndexFromState(finalState));
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
    await revertStepInBackend({
      to_before_retries: toBeforeRetries,
      workflow_id: userParams.workflow_id,
    } as RevertStepParams);
    const latestState = await getLatestState(userParams.workflowId);
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
    }
  }

  const workflowInitialized = project && latestState?.all_steps !== undefined;
  function onCancelStep() {
    throw new Error('not implemented');
  }

  return (
    <div className="flex w-full flex-col gap-4">
      <div className="flex gap-3 w-full justify-between mb-3 items-center">
        Video: {project?.video_hash}
        {workflowInitialized ? (
          <div className="flex gap-3 items-center">
            <Button variant="outline" onClick={restart} disabled={isLoading}>
              <ReloadIcon className="mr-2" />
              Restart
            </Button>
            <Button disabled={isLoading}>
              <DownloadIcon className="mr-2" />
              Export
            </Button>
          </div>
        ) : null}
      </div>

      {workflowInitialized ? (
        <StructuredInputFormProvider
          onFormDataChange={handleStructuredInputFormValueChange}
          stepOutput={stepOutput}
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
                      stepOutput={
                        latestState?.outputs?.length &&
                        latestState.outputs.length > index
                          ? latestState.outputs[index]
                          : null
                      }
                      chatMessages={chatMessages}
                      onSubmit={advanceOrRetryStep}
                      isNewStep={trueStepIndex < index}
                      isLoading={isLoading}
                      backendMessage={backendMessage}
                      isInitialized={workflowInitialized}
                      onCancelStep={onCancelStep}
                      setUserMessage={setUserMessage}
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
