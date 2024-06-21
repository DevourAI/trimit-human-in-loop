'use client';
import { DownloadIcon, ReloadIcon } from '@radix-ui/react-icons';
import React, { useEffect, useMemo, useReducer, useRef, useState } from 'react';
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
  CutTranscriptLinearWorkflowStreamingOutput,
  ExportableStepWrapper,
  GetLatestState,
} from '@/gen/openapi/api';
import {
  getLatestState,
  resetWorkflow,
  revertStepInBackend,
  revertStepToInBackend,
  step,
} from '@/lib/api';
import { decodeStreamAsJSON } from '@/lib/streams';
import {
  GetLatestStateParams,
  ResetWorkflowParams,
  RevertStepParams,
  RevertStepToParams,
  StepParams,
} from '@/lib/types';

const BASE_PROMPT = 'What do you want to create?';

function stepIndexFromState(state: GetLatestState): number {
  if (!state.all_steps) {
    throw new Error('state does not contain all_steps');
  }
  if (state && state.last_step) {
    return stepIndexFromName(state.last_step.step_name, state.all_steps);
  }
  return 0;
}

function stepIndexFromName(
  stepName: string,
  allSteps: Array<ExportableStepWrapper>
): number {
  const currentStepIndex = allSteps.findIndex((step) => step.name === stepName);
  if (currentStepIndex === -1) {
    console.error(`Could not find step ${stepName} in steps array`, allSteps);
    return 0;
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
  state: GetLatestState
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
export default function MainStepper({ videoHash }: { videoHash: string }) {
  const { userData } = useUser();
  const { stepperFormValues } = useStepperForm();
  const { toast } = useToast();
  const [latestState, setLatestState] = useState<GetLatestState | null>(null);
  const [userFeedbackRequest, setUserFeedbackRequest] = useState<string>('');
  const [trueStepIndex, setTrueStepIndex] = useState<number>(0);
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(0);
  const [latestExportResult, setLatestExportResult] = useState<
    Record<string, any>
  >({});
  const [latestExportCallId, setLatestExportCallId] = useState<string>('');
  const [stepOutput, setStepOutput] =
    useState<CutTranscriptLinearWorkflowStepOutput | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [backendMessage, setBackendMessage] = useState<string>('');
  const [currentStepFormValues, setCurrentStepFormValues] = useState<
    z.infer<typeof FormSchema>
  >({});
  const [hasCompletedAllSteps, setHasCompletedAllSteps] =
    useState<boolean>(false);

  const timelineName = 'timelineName';
  const lengthSeconds = 60;

  const userParams = useMemo(
    () => ({
      user_email: userData.email,
      timeline_name: timelineName,
      length_seconds: lengthSeconds,
      video_hash: videoHash,
    }),
    [userData.email, timelineName, lengthSeconds, videoHash]
  );
  const fetchedInitialState = useRef(false);
  const allowRunningFromCurrentStepIndexChange = useRef(false);
  const prevExportResult = useRef();
  const prevExportCallId = useRef();

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
      const data = await getLatestState(userParams as GetLatestStateParams);
      if (!data || Object.keys(data).length === 0) return;
      console.log('data', data);
      if (!fetchedInitialState.current) {
        let stepIndex = 0;
        try {
          stepIndex = stepIndexFromState(data);
          console.log('stepIndex', stepIndex);
        } catch (error) {
          console.log(error);
        }

        if (stepIndex === -1) {
          stepIndex = data.all_steps ? data.all_steps.length - 1 : 0;
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
  }, [userData, userParams, videoHash, currentStepIndex]);

  useEffect(() => {
    if (!latestState || !latestState.all_steps) return;

    const stepIndex = stepIndexFromState(latestState);
    if (stepIndex !== -1) {
      setTrueStepIndex(stepIndex);
    } else {
      setTrueStepIndex(latestState.all_steps.length - 1);
      setHasCompletedAllSteps(true);
    }

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
  }, [latestState]);

  useEffect(() => {
    setUserFeedbackRequest(stepOutput?.user_feedback_request || BASE_PROMPT);
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
    const finalState: CutTranscriptLinearWorkflowStepOutput | null =
      await decodeStreamAsJSON(
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
    setStepOutput(finalState.outputs[finalState.outputs.length - 1]);
    setIsLoading(false);
  }

  async function advanceStep(stepIndex: number) {
    setIsLoading(true);
    console.log(
      'advanceStep stepIndex',
      stepIndex,
      'trueStepIndex',
      trueStepIndex,
      'currentStepIndex',
      currentStepIndex
    );
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
    const params: StepParams = {
      user_input:
        stepperFormValues.feedback !== undefined &&
        stepperFormValues.feedback !== null
          ? stepperFormValues.feedback
          : '',
      streaming: true,
      force_restart: false,
      ignore_running_workflows: true,
      retry_step: false,
      ...userParams,
    };
    try {
      await step(params, handleStepStream);
    } catch (error) {
      console.error('error in step', error);
    }
  }

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
    const params: StepParams = {
      user_input:
        data.feedback !== undefined && data.feedback !== null
          ? data.feedback
          : '',
      streaming: true,
      force_restart: false,
      ignore_running_workflows: true,
      retry_step: true,
      ...userParams,
    };
    try {
      await step(params, handleStepStream);
    } catch (error) {
      console.error('error in step', error);
    }
  }

  async function restart() {
    setIsLoading(true);
    activePromptDispatch({ type: 'restart', value: '' });
    setStepOutput(null);
    await resetWorkflow(userParams as ResetWorkflowParams);
    const newState = await getLatestState(userParams as GetLatestStateParams);
    setLatestState(newState);
    setCurrentStepIndex(0);
    setIsLoading(false);
  }

  async function revertStep(toBeforeRetries: boolean) {
    setIsLoading(true);
    activePromptDispatch({ type: 'restart', value: '' });
    setStepOutput(null);
    await revertStepInBackend({
      to_before_retries: toBeforeRetries,
      ...userParams,
    } as RevertStepParams);
    const latestState = await getLatestState(
      userParams as GetLatestStateParams
    );
    setLatestState(latestState);
    setCurrentStepIndex(stepIndexFromState(latestState));
    setIsLoading(false);
  }

  async function revertStepTo(stepIndex: number) {
    setIsLoading(true);
    if (!latestState || !latestState.all_steps) {
      throw new Error("can't revert unless latestState.all_steps is available");
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
      const latestState = await getLatestState(
        userParams as GetLatestStateParams
      );
      setLatestState(latestState);
      setCurrentStepIndex(stepIndexFromState(latestState));
    }
    setIsLoading(false);
    return success;
  }

  async function undoLastStep() {
    await revertStep(false);
  }

  function updateStepOutput(stepIndex: number, state: GetLatestState) {
    let stepOutput: CutTranscriptLinearWorkflowStepOutput | null = null;
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
    if (currentStepIndex === 0 || isLoading || !latestState?.all_steps) {
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

  return (
    <div className="flex w-full flex-col gap-4">
      <div className="flex gap-3 w-full justify-between mb-3 items-center">
        Video: {videoHash}
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
      </div>
      {latestState?.all_steps ? (
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
                    stepOutput={
                      latestState?.outputs?.length > index
                        ? latestState.outputs[index]
                        : null
                    }
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
