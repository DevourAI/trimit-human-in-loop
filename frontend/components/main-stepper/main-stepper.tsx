'use client';
import { DownloadIcon, ReloadIcon } from '@radix-ui/react-icons';
import React, { useEffect, useMemo, useReducer, useState } from 'react';
import { z } from 'zod';

import { Footer } from '@/components/main-stepper/main-stepper-footer';
import { Button } from '@/components/ui/button';
import { Step, Stepper } from '@/components/ui/stepper';
import { FormSchema, StepperForm } from '@/components/ui/stepper-form';
import { useUser } from '@/contexts/user-context';
import {
  getLatestState,
  getStepOutput,
  resetWorkflow,
  revertStepInBackend,
  revertStepToInBackend,
  step,
} from '@/lib/api';
import { actionSteps, allSteps, stepData } from '@/lib/data';
import { decodeStreamAsJSON } from '@/lib/streams';
import {
  GetLatestStateParams,
  ResetWorkflowParams,
  RevertStepParams,
  RevertStepToParams,
  StepInfo,
  StepParams,
  UserState,
} from '@/lib/types';

const BASE_PROMPT = 'What do you want to create?';

function remove_retry_suffix(stepName: string): string {
  return stepName.split('_retry_', 1)[0];
}

function findNextActionStepIndex(
  allStepIndex: number,
  allSteps: StepInfo[],
  actionSteps: StepInfo[]
): number {
  return allStepIndex;
}

function stepIndexFromState(state: UserState): number {
  if (state && state.next_step) {
    return stepIndexFromName(
      state.next_step.step_name,
      state.next_step.name,
      allSteps,
      actionSteps
    );
  } else if (state && state.last_step) {
    return -1;
  }
  return 0;
}

function stepIndexFromName(
  stepName: string,
  substepName: string,
  allSteps: StepInfo[],
  actionSteps: StepInfo[]
): number {
  const _currentAllStepIndex = allSteps.findIndex(
    (step) =>
      step.name === remove_retry_suffix(substepName) &&
      step.step_name === stepName
  );
  if (_currentAllStepIndex !== -1) {
    const nextActionStepIndex = findNextActionStepIndex(
      _currentAllStepIndex,
      allSteps,
      actionSteps
    );
    if (nextActionStepIndex >= actionSteps.length) {
      console.log('All action steps completed');
    }
    return nextActionStepIndex;
  } else {
    console.error(
      `Could not find step ${remove_retry_suffix(substepName)} in steps array`,
      allSteps
    );
  }
  return 0;
}

function stepNameFromIndex(stepIndex: number): [string, string] {
  return [actionSteps[stepIndex].step_name, actionSteps[stepIndex].name];
}

export default function MainStepper({ videoHash }: { videoHash: string }) {
  const { userData } = useUser();
  const [latestState, setLatestState] = useState<UserState | null>(null);
  const [userFeedbackRequest, setUserFeedbackRequest] = useState<string>('');
  const [trueStepIndex, setTrueStepIndex] = useState<number>(0);
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(0);
  const [latestExportResult, setLatestExportResult] = useState<
    Record<string, any>
  >({});
  const [finalResult, setFinalResult] = useState<Record<string, any>>({});
  const [isLoading, setIsLoading] = useState<boolean>(false);
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
      let stepIndex = stepIndexFromState(data);
      if (stepIndex === -1) {
        stepIndex = actionSteps.length - 1;
      }
      setCurrentStepIndex(stepIndex);
    }

    fetchLatestState();
  }, [userData, userParams, videoHash]);

  useEffect(() => {
    if (!latestState) return;
    setUserFeedbackRequest(latestState.output?.user_feedback_request || null);
    const stepIndex = stepIndexFromState(latestState);
    if (stepIndex !== -1) {
      setTrueStepIndex(stepIndex);
    } else {
      setTrueStepIndex(actionSteps.length - 1);
      setHasCompletedAllSteps(true);
    }
  }, [latestState]);

  useEffect(() => {
    setUserFeedbackRequest(finalResult?.user_feedback_request || BASE_PROMPT);
  }, [finalResult]);

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
    const lastValue = await decodeStreamAsJSON(reader, (value: any) => {
      let valueToAppend = value;
      if (typeof value !== 'string') {
        if (value.substep_name !== undefined) {
          const stepIndex = stepIndexFromName(
            value.step_name,
            value.substep_name,
            allSteps,
            actionSteps
          );
          setTrueStepIndex(stepIndex);
          setCurrentStepIndex(stepIndex);
          setLatestExportResult(value.export_result);
          valueToAppend = '';
        } else if (value.chunk !== undefined && value.chunk === 0) {
          if (typeof value.output === 'string') {
            valueToAppend = value.output;
          } else {
            console.log('value.output is not a string', value.output);
            valueToAppend = '';
          }
        } else if (value.message !== undefined) {
          if (typeof value.message === 'string') {
            valueToAppend = value.message;
          } else if (
            value.message.output &&
            typeof value.message.output === 'string'
          ) {
            if (value.message.chunk === 0) {
              valueToAppend = value.message.output;
            } else {
              valueToAppend = '';
            }
          } else {
            console.log(
              'value.message.output is not a string',
              value.message.output
            );
            valueToAppend = '';
          }
        } else {
          valueToAppend = '';
        }
      }
      activePromptDispatch({ type: 'append', value: valueToAppend });
    });
    if (typeof lastValue !== 'string') {
      console.log('lastValue is not a string', lastValue);
    } else {
      setFinalResult(lastValue);
    }
    setIsLoading(false);
  }

  async function onSubmit(
    stepIndex: number,
    retry: boolean,
    data: z.infer<typeof FormSchema>
  ) {
    setIsLoading(true);
    if (trueStepIndex > stepIndex) {
      let success = false;
      if (retry) {
        success = await revertStepTo(stepIndex + 1);
      } else {
        success = await revertStepTo(stepIndex);
      }
      console.log('success', success);
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
      retry_step: retry,
      ...userParams,
    };
    await step(params, handleStepStream);
  }

  async function restart() {
    setIsLoading(true);
    activePromptDispatch({ type: 'restart', value: '' });
    setFinalResult({});
    await resetWorkflow(userParams as ResetWorkflowParams);
    const newState = await getLatestState(userParams as GetLatestStateParams);
    setLatestState(newState);
    setCurrentStepIndex(0);
    setIsLoading(false);
  }

  async function revertStep(toBeforeRetries: boolean) {
    setIsLoading(true);
    activePromptDispatch({ type: 'restart', value: '' });
    setFinalResult({});
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
    const [stepName, substepName] = stepNameFromIndex(stepIndex);
    console.log('stepName', stepName, 'substepName', substepName);
    const success = await revertStepToInBackend({
      step_name: stepName,
      substep_name: substepName,
      ...userParams,
    } as RevertStepToParams);
    console.log('in revertStepTo success', success);
    if (success) {
      activePromptDispatch({ type: 'restart', value: '' });
      setFinalResult({});
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

  async function undoLastStepBeforeRetries() {
    await revertStep(true);
  }

  async function prevStepWrapper() {
    if (currentStepIndex === 0) {
      return;
    }
    setCurrentStepIndex(currentStepIndex - 1);
    const currentSubstepName = actionSteps[currentStepIndex - 1].name;
    const currentStepName = actionSteps[currentStepIndex - 1].step_name;
    const currentStepKey = `${currentStepName}.${currentSubstepName}`;
    activePromptDispatch({ type: 'restart', value: '' });
    setFinalResult(
      await getStepOutput({ step_key: currentStepKey, ...userParams })
    );
  }
  async function nextStepWrapper() {
    if (currentStepIndex >= trueStepIndex) {
      return;
    }
    setCurrentStepIndex(currentStepIndex + 1);
    const currentSubstepName = actionSteps[currentStepIndex + 1].name;
    const currentStepName = actionSteps[currentStepIndex + 1].step_name;
    const currentStepKey = `${currentStepName}.${currentSubstepName}`;
    activePromptDispatch({ type: 'restart', value: '' });
    setFinalResult(
      await getStepOutput({ step_key: currentStepKey, ...userParams })
    );
  }

  return (
    <div className="flex w-full flex-col gap-4">
      <div className="flex gap-3 w-full justify-between mb-3 items-center">
        Video: {videoHash}
        <div className="flex gap-3 items-center">
          <Button variant="outline" onClick={restart}>
            <ReloadIcon className="mr-2" />
            Restart
          </Button>
          <Button>
            <DownloadIcon className="mr-2" />
            Export
          </Button>
        </div>
      </div>
      <Stepper
        initialStep={currentStepIndex}
        steps={stepData.stepArray}
        orientation="vertical"
      >
        {actionSteps.map(({ step_name, name, human_readable_name }, index) => {
          return (
            <Step key={`${step_name}.${name}`} label={human_readable_name}>
              <div className="grid w-full gap-2">
                <StepperForm
                  systemPrompt={activePrompt}
                  isLoading={isLoading}
                  prompt={userFeedbackRequest}
                  stepIndex={index}
                  onSubmit={onSubmit}
                  userParams={userParams}
                  step={actionSteps[index]}
                  onCancelStep={() => {
                    throw new Error('Unimplemented');
                  }}
                />
              </div>
            </Step>
          );
        })}
        <Footer
          currentStepIndex={currentStepIndex}
          trueStepIndex={trueStepIndex}
          prevStepWrapper={prevStepWrapper}
          nextStepWrapper={nextStepWrapper}
          undoLastStep={undoLastStep}
          hasCompletedAllSteps={hasCompletedAllSteps}
        />
      </Stepper>
    </div>
  );
}
