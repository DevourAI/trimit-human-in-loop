"use client";
import {
  Step,
  Stepper,
  useStepper,
} from "@/components/ui/stepper"
import { Button } from "@/components/ui/button"
import React, { createContext, useContext, useState, useReducer, useEffect } from 'react';
import { StepperForm, FormSchema } from "@/components/stepper-form"
import { decodeStreamAsJSON } from "@/lib/streams";
import { step, getLatestState, getStepOutput, resetWorkflow, revertStepInBackend } from "@/lib/api";
import {
  type CommonAPIParams,
  type StepOutputParams,
  type GetLatestStateParams,
  type RevertStepParams,
  type ResetWorkflowParams,
  type StepInfo,
  type UserState,
  type StepParams
} from "@/lib/types";

const BASE_PROMPT = 'What do you want to create?'

function remove_retry_suffix(stepName: string): string {
  return stepName.split('_retry_', 1)[0]
}

function stepsFromState(state: UserState): StepInfo[] {
  const defaultSteps = [
    { name: "Upload video" },
    { name: "..." },
    { name: "Profit!" },
  ] satisfies StepInfo[]

  if (!state || !state.all_steps) {
    return [defaultSteps, defaultSteps]
  }

  const allSteps = state.all_steps.map((step) => {
    return { label: step.name, ...step }
  })
  if (allSteps.length === 0) {
    return [defaultSteps, defaultSteps]
  }
  const actionSteps = allSteps.filter((step) => step.user_feedback)
  return [allSteps, actionSteps]
}

function findNextActionStepIndex(allStepIndex, allSteps, actionSteps) {
  const actionStepNames = actionSteps.map((step) => step.name)
  console.log('actionStepNames', actionStepNames);
  for (let i = allStepIndex; i < allSteps.length; i++) {
    console.log('allStepName', allSteps[i].name);
    const actionStepIndex = actionStepNames.indexOf(allSteps[i].name)
    if (actionStepIndex > -1) {
      return actionStepIndex
    }
  }
  return actionSteps.length
}
function stepIndexFromState(state: UserState): number {
  const [allSteps, actionSteps] = stepsFromState(state)
  if (state && state.next_step) {
    const _currentAllStepIndex = allSteps.findIndex((step) => step.name === remove_retry_suffix(state.next_step.name))
    console.log('currentAllStepIndex', _currentAllStepIndex);
    if (_currentAllStepIndex !== -1) {
      const nextActionStepIndex = findNextActionStepIndex(_currentAllStepIndex, allSteps, actionSteps)
      if (nextActionStepIndex >= actionSteps.length) {
        console.log('All action steps completed');
      }
      return nextActionStepIndex
    } else {
      console.error(`Could not find step ${state.next_step.name} in steps array ${allSteps}`)
    }
  }
  return 0
}


export default function MainStepper({ userData }) {
  const videoHash = '3985222955'
  const timelineName = 'timelineName'
  const lengthSeconds = 60
  const userParams: UserParams = {
    user_email: userData.email,
    timeline_name: timelineName,
    length_seconds: lengthSeconds,
    video_hash: videoHash,
  }

  const [latestState, setLatestState] = useState(null)
  const [userFeedbackRequest, setUserFeedbackRequest] = useState(null)
  const [trueStepIndex, setTrueStepIndex] = useState(0)
  const [currentStepIndex, setCurrentStepIndex] = useState(0)
  const [finalResult, setFinalResult] = useState({})
  const [isLoading, setIsLoading] = useState(false)
  const [needsRevert, setNeedsRevert] = useState(false)
  const [allSteps, actionSteps] = stepsFromState(latestState)
  console.log('allSteps', allSteps);
  console.log('actionSteps', actionSteps);
  console.log('latestState', latestState);
  console.log('trueStepIndex', trueStepIndex);
  console.log('currentStepIndex', currentStepIndex);
  console.log('needsRevert', needsRevert);

  useEffect(() => {
    async function fetchLatestState() {
        const data = await getLatestState(userParams as GetLatestStateParams);
        console.log('Fetched _latestState', data);
        setLatestState(data);
        setCurrentStepIndex(stepIndexFromState(data))
    }

    fetchLatestState();
  }, [userData]);

  useEffect(() => {
    setUserFeedbackRequest(latestState?.output?.user_feedback_request)
    setTrueStepIndex(stepIndexFromState(latestState))

  }, [latestState]);

  useEffect(() => {
    console.log('finalResult', finalResult);
    setUserFeedbackRequest(finalResult?.user_feedback_request || BASE_PROMPT)
  }, [finalResult]);

  const [activePrompt, activePromptDispatch] = useReducer((state, action) => {
    switch (action.type) {
      case 'append':
        return state + action.value;
      case 'restart':
        return action.value;
      default:
        throw new Error('Unhandled action type');
    }
  }, '');

  async function onSubmit(stepIndex: number, data: z.infer<typeof FormSchema>) {
    setIsLoading(true)
    if (needsRevert) {
      console.log('reverting step');
      for (let i = 0; i < trueStepIndex - stepIndex; i++) {
        await undoLastStepBeforeRetries()
      }
      setNeedsRevert(false)
    }
    const params = {
      user_input: data.feedback,
      streaming: true,
      force_restart: false,
      ignore_running_workflows: true,
      ...userParams,
    } as StepParams
    await step(params, async (reader) => {
      activePromptDispatch({ type: 'restart', value: '' });
      const lastValue = await decodeStreamAsJSON(reader, (value) => {
        let valueToAppend = value;
        if (typeof value !== 'string') {
          if (value.chunk !== undefined && value.chunk === 0) {
            valueToAppend = value.output;
          }
        }
        activePromptDispatch({ type: 'append', value: valueToAppend });
      });
      setFinalResult(lastValue?.result)
      setIsLoading(false)
    })
  }



  async function restart() {
    setIsLoading(true)
    activePromptDispatch({ type: 'restart', value: '' });
    console.log('activePrompt', activePrompt);
    setFinalResult({})
    console.log('finalResult', finalResult);
    await resetWorkflow(userParams as ResetWorkflowParams)
    console.log("workflow was reset")
    const newState = await getLatestState(userParams as GetLatestStateParams)
    console.log("new state:", newState)
    setLatestState(newState)
    setCurrentStepIndex(0)
    setIsLoading(false)
  }

  async function revertStep(toBeforeRetries) {
    setIsLoading(true)
    activePromptDispatch({ type: 'restart', value: '' });
    setFinalResult({})
    await revertStepInBackend({to_before_retries: toBeforeRetries, ...userParams} as RevertStepParams)
    const latestState = await getLatestState(userParams as GetLatestStateParams)
    setLatestState(latestState)
    setCurrentStepIndex(stepIndexFromState(latestState))
    setIsLoading(false)
  }
  async function undoLastStep() {
    await revertStep(false)
  }

  async function undoLastStepBeforeRetries() {
    await revertStep(true)
  }

  async function prevStepWrapper() {
    if (currentStepIndex === 0) {
      return
    }
    setCurrentStepIndex(currentStepIndex - 1)
    const currentStepName = actionSteps[currentStepIndex].name
    activePromptDispatch({ type: 'restart', value: '' });
    setFinalResult(await getStepOutput(
      { step_keys: currentStepName, ...userParams}
    ))
    setNeedsRevert(true)
  }

  return (
    <div className="flex w-full flex-col gap-4">
      <Stepper initialStep={currentStepIndex} steps={actionSteps}>
        {actionSteps.map(({ name, input }, index) => {
          return (
            <Step key={name} label={name}>
              <div className="grid w-full gap-2">
                <StepperForm
                  systemPrompt={activePrompt}
                  undoLastStep={undoLastStep}
                  isLoading={isLoading}
                  prompt={userFeedbackRequest}
                  stepIndex={index}
                  onSubmit={onSubmit}
                  userData={userData}
                  step={actionSteps[index]}
                />
              </div>
            </Step>
          )
        })}
        <Footer currentStepIndex={currentStepIndex} prevStepWrapper={prevStepWrapper} restart={restart} />
      </Stepper>
    </div>
  )
}

const Footer = ({restart, prevStepWrapper, currentStepIndex}) => {
  const {
    nextStep,
    prevStep,
    resetSteps,
    isDisabledStep,
    hasCompletedAllSteps,
    isLastStep,
    isOptionalStep,
    setStep
  } = useStepper()
  setStep(currentStepIndex)
  return (
    <>

      {hasCompletedAllSteps && (
        <div className="h-40 flex items-center justify-center my-4 border bg-secondary text-primary rounded-md">
          <h1 className="text-xl">Woohoo! All steps completed! 🎉</h1>
        </div>
      )}
      <div className="w-full flex justify-end gap-2">
          <>
          <Button size="sm" onClick={restart}>
            Restart
          </Button>
          <Button
            disabled={currentStepIndex === 0}
            onClick={prevStepWrapper}
            size="sm"
            variant="secondary"
          >
            Prev
          </Button>
          <Button size="sm" onClick={nextStep}>
            {isLastStep ? "Finish" : isOptionalStep ? "Skip" : "Next"}
          </Button>
          </>
      </div>
    </>
  )
}

// TODO We show next step too early. It should happen after first part of output comes through.
// Or maybe backend should notify that we are moving on to the next step
// When we revert, we end up doing steps twice somehow.
// setStep showing errors
//  // TODOs: split chunks on UI

