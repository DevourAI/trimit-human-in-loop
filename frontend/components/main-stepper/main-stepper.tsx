'use client';
import { DownloadIcon, ReloadIcon } from '@radix-ui/react-icons';
import React, { useEffect, useMemo, useReducer, useState } from 'react';
import { z } from 'zod';

import { Footer } from '@/components/main-stepper/main-stepper-footer';
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
  getStepOutput,
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

  useEffect(() => {
    async function fetchLatestState() {
      const data = await getLatestState(userParams as GetLatestStateParams);
      setLatestState(data);
      console.log('data', data);
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
    }

    fetchLatestState();
  }, [userData, userParams, videoHash, currentStepIndex]);

  useEffect(() => {
    if (!latestState || !latestState.all_steps) return;
    setUserFeedbackRequest(latestState.output?.user_feedback_request || '');
    const stepIndex = stepIndexFromState(latestState);
    if (stepIndex !== -1) {
      setTrueStepIndex(stepIndex);
    } else {
      setTrueStepIndex(latestState.all_steps.length - 1);
      setHasCompletedAllSteps(true);
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
    const lastValue: CutTranscriptLinearWorkflowStepOutput | null =
      await decodeStreamAsJSON(
        reader,
        (value: CutTranscriptLinearWorkflowStreamingOutput) => {
          setIsLoading(false);
          if (value?.partial_step_output) {
            let output = value.partial_step_output;
            console.log('partial step output', output);
            if (latestState?.all_steps) {
              const stepIndex = stepIndexFromName(
                output.step_name,
                latestState.all_steps
              );
              if (stepIndex !== trueStepIndex) {
                setTrueStepIndex(stepIndex);
              }
              if (stepIndex !== currentStepIndex) {
                setCurrentStepIndex(stepIndex);
              }
              if (
                output.export_result &&
                output.export_result != latestExportResult
              ) {
                setLatestExportResult(output.export_result);
              } else if (
                output.export_call_id &&
                output.export_call_id != latestExportCallId
              ) {
                setLatestExportCallId(output.export_call_id);
              }
            }
          } else if (value?.partial_backend_output) {
            let output = value.partial_backend_output;
            console.log('partial backend output', output);
            if (output.chunk === null || output.chunk == 0) {
              setBackendMessage(output.value || '');
            }
          } else if (value?.partial_llm_output) {
            let output = value.partial_llm_output;
            console.log('partial llm output', output);
            if (output.chunk === null || output.chunk == 0) {
              activePromptDispatch({ type: 'append', value: output.value });
              //  TODO: for some reason this is not triggering StepperForm+systemPrompt to reload
            }
          }
        }
      );
    setLatestState(lastValue);
  }

  async function advanceStep(stepIndex: number) {
    setIsLoading(true);
    if (trueStepIndex > stepIndex) {
      // TODO send name of step and have backend to reversions
      const success = await revertStepTo(stepIndex);
      if (!success) {
        setIsLoading(false);
        return;
      }
    }
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

  async function onPrevStep() {
    if (currentStepIndex === 0 || isLoading || !latestState?.all_steps) {
      return;
    }
    setCurrentStepIndex(currentStepIndex - 1);
    const currentStepName = latestState.all_steps[currentStepIndex - 1].name;
    activePromptDispatch({ type: 'restart', value: '' });
    setStepOutput(
      await getStepOutput({ step_name: currentStepName, ...userParams })
    );
  }
  async function onNextStep() {
    if (
      isLoading ||
      !latestState?.all_steps ||
      currentStepIndex > trueStepIndex
    ) {
      return;
    }
    setCurrentStepIndex(currentStepIndex + 1);
    const currentStepName = latestState.all_steps[currentStepIndex + 1].name;
    activePromptDispatch({ type: 'restart', value: '' });
    if (currentStepIndex == trueStepIndex) {
      await advanceStep(currentStepIndex);
    } else {
      setStepOutput(
        await getStepOutput({ step_name: currentStepName, ...userParams })
      );
    }
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
          steps={latestState.all_steps.map((step) => {
            return { label: step.human_readable_name || step.name };
          })}
          orientation="vertical"
        >
          {latestState.all_steps.map((step, index) => (
            <Step key={step.name} label={step.human_readable_name || step.name}>
              <div className="grid w-full gap-2">
                <StepperForm
                  systemPrompt={activePrompt}
                  isLoading={isLoading}
                  prompt={userFeedbackRequest}
                  stepIndex={index}
                  onRetry={retryStep}
                  userParams={userParams}
                  step={step}
                  onCancelStep={() => {
                    throw new Error('Unimplemented');
                  }}
                />

                {/* <StepRenderer
                  step={step}
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
                /> */}
              </div>
              <Footer
                currentStepIndex={currentStepIndex}
                trueStepIndex={trueStepIndex}
                onPrevStep={onPrevStep}
                onNextStep={onNextStep}
                undoLastStep={undoLastStep}
                hasCompletedAllSteps={hasCompletedAllSteps}
                totalNSteps={latestState.all_steps!.length}
              />
            </Step>
          ))}
        </Stepper>
      ) : null}
    </div>
  );
}
