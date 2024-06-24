'use client';
import { DownloadIcon, ReloadIcon } from '@radix-ui/react-icons';
import React, {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from 'react';
import { z } from 'zod';

import { Footer } from '@/components/main-stepper/main-stepper-footer';
import StepRenderer from '@/components/main-stepper/step-renderer';
import { Button } from '@/components/ui/button';
import { Step, Stepper } from '@/components/ui/stepper';
import { FormSchema, StepperForm } from '@/components/ui/stepper-form';
import { useToast } from '@/components/ui/use-toast';
import { useStepperForm } from '@/contexts/stepper-form-context';
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
import {
  ResetWorkflowParams,
  RevertStepParams,
  RevertStepToParams,
} from '@/lib/types';

function stepIndexFromState(state: FrontendWorkflowState): number {
  if (!state.all_steps) {
    throw new Error('state does not contain all_steps');
  }
  if (state && state.outputs && state.outputs.length) {
    return stepIndexFromName(
      state.outputs[state.outputs.length - 1].step_name,
      state.all_steps
    );
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
  const { toast } = useToast();
  const [latestState, setLatestState] = useState<FrontendWorkflowState | null>(
    null
  );
  const [userFeedbackRequest, setUserFeedbackRequest] = useState<string>('');
  const [stepInputPrompt, setStepInputPrompt] = useState<string>('');
  const [trueStepIndex, setTrueStepIndex] = useState<number>(-1);
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(-1);
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
      console.log('fetchedInitialState.current', fetchedInitialState.current);
      console.log(
        'allowRunningFromCurrentStepIndexChange.current',
        allowRunningFromCurrentStepIndexChange.current
      );
      if (
        fetchedInitialState.current &&
        !allowRunningFromCurrentStepIndexChange.current
      ) {
        allowRunningFromCurrentStepIndexChange.current = true;
        return;
      }
      const data = await getLatestState(userParams.workflow_id);
      if (!data || Object.keys(data).length === 0) return;
      console.log('data', data);
      if (!fetchedInitialState.current) {
        let stepIndex = -1;
        try {
          stepIndex = stepIndexFromState(data);
          console.log('stepIndex', stepIndex);
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
  useEffect(() => {
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

    if (!latestState.outputs || !latestState.outputs.length) return;
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
  }, [latestState, currentStepIndex]);

  useEffect(() => {
    setUserFeedbackRequest(stepOutput?.user_feedback_request || '');
  }, [stepOutput]);

  type ActivePromptAction =
    | { type: 'append'; value: string }
    | { type: 'restart'; value: string };

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

  async function handleStepStream(reader: ReadableStreamDefaultReader) {
    activePromptDispatch({ type: 'restart', value: '' });
    const finalState: FrontendWorkflowState | null = await decodeStreamAsJSON(
      reader,
      (value: CutTranscriptLinearWorkflowStreamingOutput) => {
        if (value?.partial_step_output) {
          // we can log this maybe
        } else if (value?.partial_backend_output) {
          let output = value.partial_backend_output;
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
  }

  const revertStepTo = useCallback(
    async (stepIndex: number) => {
      setIsLoading(true);
      if (!latestState || !latestState.all_steps) {
        throw new Error(
          "can't revert unless latestState.all_steps is available"
        );
      }
      const stepName = stepNameFromIndex(latestState.all_steps, stepIndex);
      console.log('stepName', stepName);
      const success = await revertStepToInBackend({
        step_name: stepName,
        ...userParams,
      } as RevertStepToParams);
      console.log('in revertStepTo success', success);
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

  const advanceStep = useCallback(
    async (stepIndex: number) => {
      setIsLoading(true);
      if (trueStepIndex >= stepIndex) {
        console.log('reverting step to before', stepIndex);
        // TODO send name of step and have backend do reversions
        const success = await revertStepTo(stepIndex);
        console.log('revert success', success);
        if (!success) {
          setIsLoading(false);
          return;
        }
      }
      setCurrentStepIndex(stepIndex);
      setStepOutput(null);
      const data: StepData = {
        user_input:
          stepperFormValues.feedback !== undefined &&
          stepperFormValues.feedback !== null
            ? stepperFormValues.feedback
            : '',
        streaming: true,
        force_restart: false,
        ignore_running_workflows: true,
        retry_step: false,
      };
      try {
        await step(userParams.workflow_id, data, handleStepStream);
      } catch (error) {
        console.error('error in step', error);
      }
    },
    [trueStepIndex, userParams, revertStepTo, stepperFormValues]
  );

  useEffect(() => {
    if (project && !latestState) {
      advanceStep(0);
    }
  }, [project, latestState, advanceStep]);

  async function retryStep(
    stepIndex: number,
    data: z.infer<typeof FormSchema>
  ) {
    console.log('passed form data', data);
    console.log('form context data', stepperFormValues);
    // TODO: stepperFormValues.feedback is cut off- doesn't include last character
    setIsLoading(true);
    if (trueStepIndex > stepIndex) {
      const success = await revertStepTo(stepIndex + 1);
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
  async function retryStepNoForm(
    stepIndex: number,
    userMessage: string,
    callback: (aiMessage: string) => void
  ) {
    // TODO: stepperFormValues.feedback is cut off- doesn't include last character
    setIsLoading(true);
    if (trueStepIndex > stepIndex) {
      const success = await revertStepTo(stepIndex + 1);
      if (!success) {
        setIsLoading(false);
        return;
      }
    }
    const stepData: StepData = {
      user_input: userMessage,
      streaming: true,
      force_restart: false,
      ignore_running_workflows: true,
      retry_step: true,
    };
    try {
      await step(stepParams, stepData, async (reader) => {
        const finalState = await handleStepStream(reader);
        const stepOutput = finalState?.outputs?.length
          ? finalState.outputs[finalState.outputs.length - 1]
          : null;
        const aiMessage = stepOutput?.full_conversation?.length
          ? stepOutput.full_conversation[
              stepOutput.full_conversation.length - 1
            ].value
          : '';
        callback(aiMessage);
      });
    } catch (error) {
      console.error('error in step', error);
    }
  }

  async function restart() {
    setIsLoading(true);
    activePromptDispatch({ type: 'restart', value: '' });
    setStepOutput(null);
    await resetWorkflow(stepParams as ResetWorkflowParams);
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
      ...stepParams,
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
    console.log(
      'onPrevStep trueStepIndex',
      trueStepIndex,
      'currentStepIndex',
      currentStepIndex
    );
    if (currentStepIndex === -1 || isLoading || !latestState?.all_steps) {
      return;
    }
    setCurrentStepIndex(currentStepIndex - 1);
    activePromptDispatch({ type: 'restart', value: '' });
    console.log('onPrevStep updating step output to ', currentStepIndex - 1);
    updateStepOutput(currentStepIndex - 1, latestState);
  }
  async function onNextStep() {
    console.log(
      'onNextStep trueStepIndex',
      trueStepIndex,
      'currentStepIndex',
      currentStepIndex
    );
    if (
      isLoading ||
      !latestState?.all_steps ||
      currentStepIndex >= trueStepIndex
    ) {
      return;
    }
    if (currentStepIndex > trueStepIndex) {
      setCurrentStepIndex(trueStepIndex);
      return;
    }
    setCurrentStepIndex(currentStepIndex + 1);
    // TODO: When the backend is modified to return conversations per-step,
    // we should update the conversation messages here
    // (currently activePromptDispatch but kasp is adding better conversation UX)
    // using the messages in latestState for this particular step
    activePromptDispatch({ type: 'restart', value: '' });
    console.log('onNextStep updating step output to ', currentStepIndex + 1);
    updateStepOutput(currentStepIndex + 1, latestState);
  }

  useEffect(() => {
    if (!backendMessage) return;
    toast({
      title: backendMessage,
      // description: "Friday, February 10, 2023 at 5:57 PM",
      // action: (
      // <ToastAction altText="Goto schedule to undo">Undo</ToastAction>
      // ),
    });
  }, [backendMessage, toast]);

  const workflowInitialized = project && latestState?.all_steps !== undefined;
  console.log('project', project);
  console.log('workflowInitialized', workflowInitialized);
  console.log('latestState', latestState);
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
        <Stepper
          initialStep={currentStepIndex}
          steps={latestState.all_steps.map((step: ExportableStepWrapper) => {
            return { label: step.human_readable_name || step.name };
          })}
          orientation="vertical"
        >
          {latestState.all_steps.map(
            (step: ExportableStepWrapper, index: number) => (
              <Step
                key={step.name}
                label={step.human_readable_name || step.name}
              >
                <div className="grid w-full gap-2">
                  <StepperForm
                    systemPrompt={activePrompt}
                    isInitialized={workflowInitialized}
                    isLoading={isLoading}
                    prompt={userFeedbackRequest}
                    stepIndex={index}
                    onRetry={retryStep}
                    onSubmit={advanceStep}
                    userParams={userParams}
                    step={step}
                    onCancelStep={() => {
                      throw new Error('Unimplemented');
                    }}
                  />

                  <StepRenderer
                    step={step}
                    stepIndex={index}
                    stepInputPrompt={stepInputPrompt}
                    stepOutput={
                      latestState?.outputs?.length &&
                      latestState.outputs.length > index
                        ? latestState.outputs[index]
                        : null
                    }
                    onRetry={retryStepNoForm}
                    footer={
                      <Footer
                        currentStepIndex={currentStepIndex}
                        trueStepIndex={trueStepIndex}
                        onPrevStep={onPrevStep}
                        onNextStep={onNextStep}
                        undoLastStep={undoLastStep}
                        hasCompletedAllSteps={hasCompletedAllSteps}
                        totalNSteps={latestState.all_steps!.length}
                      />
                    }
                  />
                </div>
                {/* <Footer
                  currentStepIndex={currentStepIndex}
                  trueStepIndex={trueStepIndex}
                  onPrevStep={onPrevStep}
                  onNextStep={onNextStep}
                  undoLastStep={undoLastStep}
                  hasCompletedAllSteps={hasCompletedAllSteps}
                  totalNSteps={latestState.all_steps!.length}
                /> */}
              </Step>
            )
          )}
        </Stepper>
      ) : null}
    </div>
  );
}
