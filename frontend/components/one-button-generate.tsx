'use client';
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

import Chat from '@/components/chat/chat';
import { Message } from '@/components/chat/chat';
import StepOutput from '@/components/main-stepper/step-output';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardFooter } from '@/components/ui/card';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import ExportStepMenu from '@/components/ui/export-step-menu';
import { Heading } from '@/components/ui/heading';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import VideoSelector from '@/components/ui/video-selector';
import { StructuredInputFormProvider } from '@/contexts/structured-input-form-context';
import { useUser } from '@/contexts/user-context';
import {
  CutTranscriptLinearWorkflowStreamingOutput,
  FrontendStepOutput,
  FrontendWorkflowState,
} from '@/gen/openapi/api';
import {
  getLatestState,
  getWorkflowDetails,
  LocalRunInput,
  run,
} from '@/lib/api';
import { decodeStreamAsJSON } from '@/lib/streams';
const POLL_INTERVAL = 5000;

const videoTypeExamples = [
  'Customer testimonial',
  'Promotional sales',
  'Product review',
  'Travel vlog',
  'Interview highlights',
];
export default function OneButtonGenerate({
  initialWorkflowId,
  initialProjectName,
  initialProjectId,
}: {
  initialWorkflowId: string | null;
  initialProjectName: string | null;
  initialProjectId: string | null;
}) {
  const { userData } = useUser();
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [stepOutput, setStepOutput] = useState<FrontendStepOutput | null>(null);
  const [mappedExportResult, setMappedExportResult] = useState<Record<
    string,
    any
  > | null>(null);
  const [userMessage, setUserMessage] = useState<string>('');
  const [backendMessage, setBackendMessage] = useState<string>('');
  const [newVideoHash, setNewVideoHash] = useState<string>('');
  const [workflowId, setWorkflowId] = useState<string>(initialWorkflowId || '');
  const [useNewWorkflowId, setUseNewWorkflowId] = useState<boolean>(false);
  const [projectId, setProjectId] = useState<string>(initialProjectId || '');
  // const [newVideoFilename, setNewVideoFilename] = useState<string>('');
  const [projectName, setProjectName] = useState<string>(
    initialProjectName || ''
  );
  const [lengthSeconds, setLengthSeconds] = useState<number | null>(120);
  const [cancelStepStream, setCancelStepStream] = useState<boolean>(false);

  const [videoType, setVideoType] = useState<string | null>(
    videoTypeExamples[0]
  );
  const [nVariations, setNVariations] = useState<number>(1);

  const [latestState, setLatestState] = useState<FrontendWorkflowState | null>(
    null
  );

  const [chatMessages, setChatMessages] = useState<Message[]>([]);

  async function fetchAndSetWorkflow(workflowId: string) {
    const workflow = await getWorkflowDetails(workflowId);
    if (workflow?.video_hash) setNewVideoHash(workflow?.video_hash);
    if (workflow?.project_id) setProjectId(workflow?.project_id);
    if (workflow?.project_name) setProjectName(workflow?.project_name);
    if (workflow?.video_type) setVideoType(workflow?.video_type);
    const state = await getLatestState(workflowId);
    if (!state || Object.keys(state).length === 0) return;
    setLatestState(state);
  }

  const isPolling = useRef<boolean>(false);
  const timeoutId = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const fetchLatestStateIfNotIsPolling = async () => {
      console.log('fetchLatestStateIfNotIsPolling');
      if (isPolling.current) return;
      console.log('no other polling running');
      if (!workflowId) return;
      console.log('found workflow, getting state');
      isPolling.current = true;
      const state = await getLatestState(workflowId);
      if (state && Object.keys(state).length === 0) {
        console.log('setting latest state');
        setLatestState(state);
      }
      isPolling.current = false;
    };
    const pollForLatestState = async () => {
      await fetchLatestStateIfNotIsPolling();
      timeoutId.current = setTimeout(pollForLatestState, POLL_INTERVAL);
    };

    if (workflowId && userData.email) {
      fetchAndSetWorkflow(workflowId);
      pollForLatestState();
    }
    return () => {
      if (timeoutId.current) {
        clearTimeout(timeoutId.current);
      }
    };
  }, [workflowId, userData.email]);

  const userParams = useMemo(() => {
    const retObj = {
      user_email: userData.email,
      workflow_id: workflowId,
      project_id: projectId,
      project_name: projectName,
      video_hash: newVideoHash,
    };
    return retObj;
  }, [userData.email, projectId, workflowId, projectName, newVideoHash]);

  const prevProjectId = useRef<string | null>(null);
  const prevNewVideoHash = useRef<string | null>(null);
  const prevLengthSeconds = useRef<number | null>(null);
  const prevStepOutput = useRef<FrontendStepOutput | null>(null);
  const prevMappedExportResult = useRef<Record<string, any> | null>(null);
  useEffect(() => {
    if (!latestState) return;
    if (latestState.outputs && latestState.outputs.length) {
      // TODO do a check here that last step is "end"
      const newStepOutput = latestState.outputs[latestState.outputs.length - 2];
      if (newStepOutput !== prevStepOutput.current) {
        setStepOutput(newStepOutput);
        prevStepOutput.current = newStepOutput;
      }
    }
    if (
      latestState.mapped_export_result &&
      latestState.mapped_export_result.length
    ) {
      const newMappedExportResult =
        latestState.mapped_export_result[
          latestState.mapped_export_result.length - 2
        ];
      if (newMappedExportResult !== prevMappedExportResult.current) {
        setMappedExportResult(newMappedExportResult);
        prevMappedExportResult.current = newMappedExportResult;
      }
    }
    if (latestState.static_state) {
      if (latestState.static_state.project_id !== prevProjectId.current) {
        setProjectName(latestState.static_state.project_name || '');
        setProjectId(latestState.static_state.project_id || '');
        prevProjectId.current = latestState.static_state.project_id || '';
      }
      if (latestState.static_state.video_hash !== prevNewVideoHash.current) {
        setNewVideoHash(latestState.static_state.video_hash);
      }
      if (
        latestState.static_state.length_seconds !== prevLengthSeconds.current
      ) {
        setLengthSeconds(latestState.static_state.length_seconds);
        prevLengthSeconds.current = latestState.static_state.length_seconds;
      }
    }
  }, [latestState]);

  const setAIMessageCallback = (aiMessage: string) => {
    setChatMessages((prevMessages) => [
      ...prevMessages,
      { sender: 'AI', text: aiMessage },
    ]);
  };

  const oldWorkflowId = useRef<string | null>(workflowId);
  const handleStepStream = useCallback(
    async (reader: ReadableStreamDefaultReader) => {
      const finalState: FrontendWorkflowState | null = await decodeStreamAsJSON(
        reader,
        (value: CutTranscriptLinearWorkflowStreamingOutput) => {
          if (cancelStepStream) {
            setCancelStepStream(false);
            reader.cancel();
            return;
          }
          if (
            value &&
            value.workflow_id &&
            value.workflow_id !== oldWorkflowId.current
          ) {
            setWorkflowId(value.workflow_id);
            oldWorkflowId.current = value.workflow_id;
          }
          if (value?.partial_step_output) {
            // we can log this maybe
          } else if (value?.partial_backend_output) {
            let output = value.partial_backend_output;
            if (output.chunk === null || output.chunk == 0) {
              setBackendMessage(output.value || '');
            }
          } else if (value?.partial_llm_output) {
          }
        }
      );
      if (cancelStepStream) {
        setCancelStepStream(false);
        return;
      }
      if (finalState?.id) setWorkflowId(finalState.id);
      setIsLoading(false);
      setLatestState(finalState);
      return finalState;
    },
    [cancelStepStream]
  );

  async function runWorkflow() {
    setIsLoading(true);
    setStepOutput(null);
    setMappedExportResult(null);
    if (newVideoHash === '') {
      return;
    }
    const runData: LocalRunInput = {
      user_input: userMessage || '',
      streaming: true,
      ignore_running_workflows: true,
      video_hash: newVideoHash,
      length_seconds: lengthSeconds,
      n_variations: nVariations,
      // timeline_name: timelineName, TODO option for this once I figure out the UI
      video_type: videoType,
      user_email: userParams.user_email,
      project_id: userParams.project_id,
      project_name: userParams.project_name,
      workflow_id: useNewWorkflowId ? '' : userParams.workflow_id,
    };
    try {
      await run(runData, async (reader) => {
        const finalState = await handleStepStream(reader);
        // TODO change backend to not send "end" state
        const _stepOutput = finalState?.outputs?.length
          ? finalState.outputs[finalState.outputs.length - 2]
          : null;
        const aiMessage = _stepOutput?.full_conversation?.length
          ? _stepOutput.full_conversation[
              _stepOutput.full_conversation.length - 1
            ].value
          : '';
        setAIMessageCallback(aiMessage);
      });
    } catch (error) {
      console.error('error in run', error);
    }
  }

  let stepName = '';
  if (latestState?.all_steps) {
    stepName =
      latestState?.all_steps[latestState.all_steps.length - 1].name || '';
  }

  const handleVideoSelected = (hash: string, filename: string) => {
    if (hash !== newVideoHash) {
      setNewVideoHash(hash);
      //setNewVideoFilename(filename);
      if (newVideoHash !== '') {
        setUseNewWorkflowId(true);
      }
    }
  };

  const restart = () => {
    setUseNewWorkflowId(true);
    setStepOutput(null);
    setMappedExportResult(null);
    setBackendMessage('');
    setChatMessages([]);
    setLatestState(null);
    setMappedExportResult(null);
    setCancelStepStream(true);
    setIsLoading(false);
  };

  return (
    <StructuredInputFormProvider
      onFormDataChange={() => {}}
      stepOutput={stepOutput}
      userParams={userParams}
      mappedExportResult={mappedExportResult}
    >
      <div className="flex w-full flex-col gap-4">
        <VideoSelector setVideoDetails={handleVideoSelected} />

        <Card className="max-w-full shadow-none">
          <CardContent className="flex max-w-full p-0">
            <div className="w-1/2 p-4">
              <Label htmlFor="lengthSeconds">
                Desired length of video (seconds)
              </Label>
            </div>
            <div className="w-1/2 border-l p-4">
              <Input
                id="lengthSeconds"
                value={lengthSeconds || ''}
                onChange={(e) => {
                  setUseNewWorkflowId(true);
                  setLengthSeconds(
                    e.target.value ? parseFloat(e.target.value) : null
                  );
                }}
              />
            </div>
          </CardContent>
        </Card>
        <Card className="max-w-full shadow-none">
          <CardContent className="flex max-w-full p-0">
            <div className="w-1/2 p-4">
              <Label htmlFor="videoType">Video type</Label>
            </div>
            <div className="w-1/2 border-l p-4">
              <div className="flex items-center">
                <Input
                  id="videoType"
                  value={videoType || ''}
                  onChange={(e) => {
                    setUseNewWorkflowId(true);
                    setVideoType(e.target.value);
                  }}
                />

                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline">Examples</Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent>
                    {videoTypeExamples.map((videoType: string) => (
                      <DropdownMenuItem
                        key={videoType}
                        onSelect={() => setVideoType(videoType)}
                      >
                        {videoType}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="max-w-full shadow-none">
          <CardContent className="flex max-w-full p-0">
            <div className="w-1/2 p-4">
              <Label htmlFor="workflowName">Project name (optional)</Label>
            </div>
            <div className="w-1/2 border-l p-4">
              <Input
                id="projectName"
                value={projectName || ''}
                onChange={(e) => {
                  setUseNewWorkflowId(true);
                  setProjectName(e.target.value);
                }}
              />
            </div>
          </CardContent>
        </Card>

        <Card className="max-w-full shadow-none">
          <CardContent className="flex max-w-full p-0">
            <div className="w-1/2 p-4">
              <Label htmlFor="nVariations">Number of variations</Label>
            </div>
            <div className="w-1/2 border-l p-4">
              <Input
                id="nVariations"
                value={nVariations}
                onChange={(e) => {
                  setNVariations(e.target.value ? parseInt(e.target.value) : 1);
                }}
              />
            </div>
          </CardContent>
        </Card>
        <Card className="max-w-full shadow-none">
          <CardContent className="flex max-w-full p-0">
            <div className="w-1/2 p-4">
              <Heading className="mb-3" size="sm">
                Chat
              </Heading>
              <Chat
                onSubmit={runWorkflow}
                disabled={!newVideoHash}
                disabledMessage={'Must select a video first'}
                isLoading={isLoading}
                messages={chatMessages}
                onChange={(userMessage: string) => {
                  setUserMessage(userMessage);
                }}
                userMessage={userMessage}
              />
            </div>
            <div className="w-1/2 border-l p-4">
              <Heading className="mb-3" size="sm">
                Outputs
              </Heading>
              {stepOutput ? (
                <div>
                  <StepOutput
                    onSubmit={() => {}}
                    isLoading={isLoading}
                    output={stepOutput}
                    allowModification={false}
                    exportCallId={null}
                    exportResult={null}
                  />
                  <ExportStepMenu
                    disabled={false}
                    userParams={userParams}
                    stepName={stepName}
                  />
                </div>
              ) : null}
            </div>
          </CardContent>

          <CardFooter>
            {isLoading && (
              <div className="w-full bg-background/90 flex justify-center items-center flex-col gap-3 text-sm">
                <Heading size="sm">
                  We&apos;ll email you when the video is done.
                </Heading>
                <Button variant="default" onClick={restart}>
                  Start Over
                </Button>
                {`${backendMessage || 'Interacting with AI'}...`}
                <LoadingSpinner size="large" />
              </div>
            )}
          </CardFooter>
        </Card>
      </div>
    </StructuredInputFormProvider>
  );
}
