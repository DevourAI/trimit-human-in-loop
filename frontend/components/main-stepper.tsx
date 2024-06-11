"use client";
import {
  Step,
  Stepper,
  useStepper,
} from "@/components/ui/stepper"
import { Button } from "@/components/ui/button"
import React, { createContext, useContext, useState, useReducer, useEffect } from 'react';
import { StepperForm, FormSchema } from "@/components/stepper-form"
import UploadVideo from "@/components/ui/upload-video"
import VideoSelector from "@/components/ui/video-selector"
import { decodeStreamAsJSON } from "@/lib/streams";
import {
  step,
  getLatestState,
  getStepOutput,
  resetWorkflow,
  revertStepInBackend,
  uploadVideo,
  downloadVideo,
  downloadTimeline,
  downloadTranscriptText,
  downloadSoundbitesText,
  getVideoProcessingStatuses,
} from "@/lib/api";
import {
  type CommonAPIParams,
  type StepOutputParams,
  type GetLatestStateParams,
  type RevertStepParams,
  type ResetWorkflowParams,
  type StepInfo,
  type UserState,
  type StepParams,
  type DownloadVideoParams
} from "@/lib/types";
import { stepData, allSteps, actionSteps } from "@/lib/data";

const BASE_PROMPT = 'What do you want to create?'
const POLL_INTERVAL = 5000

function remove_retry_suffix(stepName: string): string {
  return stepName.split('_retry_', 1)[0]
}


function findNextActionStepIndex(allStepIndex, allSteps, actionSteps) {
  // TODO remove action steps entirely, not needed anymore. The folowing is a stub until the notion is removed from code
  return allStepIndex
}

function stepIndexFromState(state: UserState): number {
  if (state && state.next_step) {
    return stepIndexFromName(state.next_step.step_name, state.next_step.name, allSteps, actionSteps)
  } else if (state && state.last_step) {
    // next step is none if we have finished , but we can stay on last step for retry handling
    // TODO: something on the UI that signals we've finished
    return -1
  }
  return 0
}

function stepIndexFromName(stepName: string, substepName: string, allSteps: StepInfo[], actionSteps: StepInfo[]): number {
  const _currentAllStepIndex = allSteps.findIndex((step) => (step.name === remove_retry_suffix(substepName) && step.step_name == stepName))
  console.log('stepName', stepName, 'substepName', substepName, 'allSteps', allSteps, 'actionSteps', actionSteps, '_currentAllStepIndex', _currentAllStepIndex)
  if (_currentAllStepIndex !== -1) {
    const nextActionStepIndex = findNextActionStepIndex(_currentAllStepIndex, allSteps, actionSteps)
    if (nextActionStepIndex >= actionSteps.length) {
      console.log('All action steps completed');
    }
    return nextActionStepIndex
  } else {
    console.error(`Could not find step ${remove_retry_suffix(stepName)} in steps array ${allSteps}`)
  }
  return 0
}


export default function MainStepper({ userData }) {
  const [videoHash, setVideoHash] = useState(null)
  const [videoProcessingStatuses, setVideoProcessingStatuses] = useState({})
  const [latestState, setLatestState] = useState(null)
  const [userFeedbackRequest, setUserFeedbackRequest] = useState(null)
  const [trueStepIndex, setTrueStepIndex] = useState(0)
  const [currentStepIndex, setCurrentStepIndex] = useState(0)
  const [latestExportResult, setLatestExportResult] = useState({})
  const [finalResult, setFinalResult] = useState({})
  const [isLoading, setIsLoading] = useState(false)
  const [needsRevert, setNeedsRevert] = useState(false)
  const [hasCompletedAllSteps, setHasCompletedAllSteps] = useState(false)

  const timelineName = 'timelineName'
  const lengthSeconds = 60
  const userParams: UserParams = {
    user_email: userData.email,
    timeline_name: timelineName,
    length_seconds: lengthSeconds,
    video_hash: videoHash,
  }


  useEffect(() => {
    async function fetchLatestState() {
        const data = await getLatestState(userParams as GetLatestStateParams);
        setLatestState(data);
        console.log('data', data)
        let stepIndex = stepIndexFromState(data)
        if (stepIndex === -1) {
          stepIndex = actionSteps.length - 1
        }
        setCurrentStepIndex(stepIndex)
        if (userData.email) {
          const videoProcessesStatusesRaw = await getVideoProcessingStatuses(userData.email);
          const newVideoProcessingStatuses = {...videoProcessingStatuses}
          if (videoProcessesStatusesRaw.result && videoProcessesStatusesRaw.result !== "error") {
            videoProcessesStatusesRaw.result.forEach((result) => {
              newVideoProcessingStatuses[result.video_hash] = {
                status: result.status,
              }
            })
          }
          setVideoProcessingStatuses(newVideoProcessingStatuses)
        }

    }

    fetchLatestState();
  }, [userData, videoHash]);

  useEffect(() => {
    setUserFeedbackRequest(latestState?.output?.user_feedback_request)
    const stepIndex = stepIndexFromState(latestState)
    if (stepIndex !== -1) {
      setTrueStepIndex(stepIndex)
    } else {
      setTrueStepIndex(actionSteps.length - 1)
      setHasCompletedAllSteps(true)
    }
  }, [latestState]);

  useEffect(() => {
    setUserFeedbackRequest(finalResult?.user_feedback_request || BASE_PROMPT)
  }, [finalResult]);

  useEffect(() => {
    let timeoutId;
    async function pollForDone() {
      const data = await getVideoProcessingStatuses(userData.email);
      let anyPending = false
      if (data.result && data.result !== "error") {
        const newVideoProcessingStatuses = {...videoProcessingStatuses}
        data.result.forEach((result) => {
          const videoHash = result.video_hash
          if (result.status === "done") {
            delete newVideoProcessingStatuses[videoHash]
          } else if (result.status === "error") {
            console.error(`Error processing video ${videoHash}: ${result.error}`)
            newVideoProcessingStatuses[videoHash].status = "error"
          } else {
            anyPending = true
          }
        })
        setVideoProcessingStatuses(newVideoProcessingStatuses)
      }
      if (anyPending) {
        timeoutId = setTimeout(pollForDone, POLL_INTERVAL);
      }
    }

    if (Object.keys(videoProcessingStatuses).length) {
      pollForDone();
    }

    // Clean up the timeout on component unmount
    return () => clearTimeout(timeoutId);
  }, [videoProcessingStatuses]);

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
      for (let i = 0; i < trueStepIndex - stepIndex; i++) {
        await undoLastStepBeforeRetries()
      }
      setNeedsRevert(false)
    }
    const params = {
      user_input: data.feedback || '',
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
          if (value.substep_name !== undefined) {
            const stepIndex = stepIndexFromName(value.substep_name, allSteps, actionSteps)
            setTrueStepIndex(stepIndex)
            setCurrentStepIndex(stepIndex)
            setLatestExportResult(value.export_result)
            valueToAppend = ''
          } else if (value.chunk !== undefined && value.chunk === 0) {
            if (typeof value.output === 'string') {
              valueToAppend = value.output;
            } else {
              console.log('value.output is not a string', value.output)
              valueToAppend = ''
            }
          } else if (value.message !== undefined) {
            if (typeof value.message === 'string') {
              valueToAppend = value.message;
            } else if (value.message.output && typeof value.message.output === 'string') {
              if (value.message.chunk === 0) {
                valueToAppend = value.message.output
              } else {
                valueToAppend = ''
              }
            } else {
              console.log('value.message.output is not a string', value.message.output)
              valueToAppend = ''
            }
          } else {
            valueToAppend = ''
          }
        }
        activePromptDispatch({ type: 'append', value: valueToAppend });
      });
      if (!typeof lastValue === 'string') {
        console.log('lastValue is not a string', lastValue)
      } else {
        setFinalResult(lastValue)
      }
      setIsLoading(false)
    })
  }



  async function restart() {
    setIsLoading(true)
    activePromptDispatch({ type: 'restart', value: '' });
    setFinalResult({})
    await resetWorkflow(userParams as ResetWorkflowParams)
    const newState = await getLatestState(userParams as GetLatestStateParams)
    setLatestState(newState)
    setCurrentStepIndex(0)
    setIsLoading(false)
  }

  async function revertStep(toBeforeRetries) {
    // revertStepInBackend should take a step name or index as a parameter and do multiple reverts if needed
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
    const currentSubstepName = actionSteps[currentStepIndex].name
    const currentStepName = actionSteps[currentStepIndex].step_name
    const currentStepKey = `${currentStepName}.${currentSubstepName}`
    activePromptDispatch({ type: 'restart', value: '' });
    setFinalResult(await getStepOutput(
      { step_key: currentStepKey, ...userParams}
    ))
    setNeedsRevert(true)
  }

  async function uploadVideoWrapper(videoFile) {
    const respData = await uploadVideo({videoFile, userEmail: userData.email, timelineName})
    if (respData && respData.processing_call_id) {
      const newEntries = {
        [respData.video_hashes[0]]: {
          callId: respData.processing_call_id,
          status: "pending"
        }
      }

      setVideoProcessingStatuses({...videoProcessingStatuses, ...newEntries})
    }
  }

  const downloadParams = {user_email: userData.email, timeline_name: timelineName, video_hash: videoHash, length_seconds: lengthSeconds}

  return (
    <div className="flex w-full flex-col gap-4">
       <UploadVideo uploadVideo={uploadVideoWrapper}/>
       <VideoSelector userData={userData} videoProcessingStatuses={videoProcessingStatuses} setVideoHash={setVideoHash}/>
       <Stepper initialStep={currentStepIndex} steps={stepData.stepArray}>
         {actionSteps.map(({ step_name, name, human_readable_name }, index) => {
           return (
             <Step key={`${step_name}.${name}`} label={human_readable_name}>
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
       <Footer
         currentStepIndex={currentStepIndex}
         prevStepWrapper={prevStepWrapper}
         restart={restart}
         hasCompletedAllSteps={hasCompletedAllSteps}
       />
       </Stepper>
       <Button onClick={() => downloadVideo(downloadParams)}>
          Download latest video
       </Button>
       <Button onClick={() => downloadTimeline(downloadParams)}>
          Download latest timeline
       </Button>
       <Button onClick={() => downloadTranscriptText(downloadParams)}>
          Download latest transcript
       </Button>
       <Button onClick={() => downloadSoundbitesText(downloadParams)}>
          Download latest soundbites transcript
       </Button>
    </div>
  )
}

const Footer = ({restart, prevStepWrapper, currentStepIndex, hasCompletedAllSteps}) => {
  const {
    nextStep,
    prevStep,
    resetSteps,
    isDisabledStep,
    isLastStep,
    isOptionalStep,
    setStep
  } = useStepper()
  useEffect(() => {
    setStep(currentStepIndex);
  }, [currentStepIndex]);

  return (
    <>

      {hasCompletedAllSteps && (
        <div className="h-40 flex items-center justify-center my-4 border bg-secondary text-primary rounded-md">
          <h3 className="text-xl">TrimIt finished editing your video, but feel free to provide additional feedback or go back to previous steps</h3>
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


