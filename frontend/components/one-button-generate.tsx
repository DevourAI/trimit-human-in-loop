'use client';
import React, { useCallback, useEffect, useMemo, useState } from 'react';

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
  FrontendWorkflowProjection,
  FrontendWorkflowState,
  RunInput,
} from '@/gen/openapi/api';
import { getLatestState, getWorkflowDetails, run } from '@/lib/api';
import { decodeStreamAsJSON } from '@/lib/streams';

const videoTypeExamples = [
  'Customer testimonial',
  'Promotional sales',
  'Product review',
  'Travel vlog',
];
export default function OneButtonGenerate({
  projectId,
}: {
  projectId: string | null;
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
  const [workflowId, setWorkflowId] = useState<string>('');
  const [newVideoFilename, setNewVideoFilename] = useState<string>('');
  const [projectName, setProjectName] = useState<string>('');
  const [lengthSeconds, setLengthSeconds] = useState<number | null>(120);
  const [cancelStepStream, setCancelStepStream] = useState<boolean>(false);

  const [videoType, setVideoType] = useState<string | null>(
    videoTypeExamples[0]
  );

  const [latestState, setLatestState] = useState<FrontendWorkflowState | null>(
    null
  );

  const [project, setProject] = useState<FrontendWorkflowProjection | null>(
    null
  );
  const [chatMessages, setChatMessages] = useState<Message[]>([]);

  useEffect(() => {
    if (project?.id) setWorkflowId(project?.id);
    if (project?.video_hash) setNewVideoHash(project?.video_hash);
  }, [project]);

  const userParams = useMemo(() => {
    return {
      user_email: userData.email,
      workflow_id: workflowId,
      video_hash: newVideoHash,
    };
  }, [userData.email, workflowId, newVideoHash]);

  useEffect(() => {
    if (!latestState) return;
    if (latestState.outputs && latestState.outputs.length) {
      // TODO do a check here that last step is "end"
      setStepOutput(latestState.outputs[latestState.outputs.length - 2]);
    }
    if (
      latestState.mapped_export_result &&
      latestState.mapped_export_result.length
    ) {
      setMappedExportResult(
        latestState.mapped_export_result[
          latestState.mapped_export_result.length - 2
        ]
      );
    }
    if (latestState.static_state) {
      setProjectName(latestState.static_state.timeline_name);
      setNewVideoHash(latestState.static_state.video_hash);
      setLengthSeconds(latestState.static_state.length_seconds);
    }
  }, [latestState]);
  useEffect(() => {
    async function fetchLatestState() {
      if (!userParams.workflow_id) return;
      const data = await getLatestState(userParams.workflow_id);
      if (!data || Object.keys(data).length === 0) return;
      setLatestState(data);
    }
    fetchLatestState();
  }, [userData, userParams]);

  const setAIMessageCallback = (aiMessage: string) => {
    setChatMessages((prevMessages) => [
      ...prevMessages,
      { sender: 'AI', text: aiMessage },
    ]);
  };

  useEffect(() => {
    async function fetchAndSetProject(projectId: string) {
      const project = await getWorkflowDetails(projectId);
      setProject(project);
    }
    if (projectId) {
      fetchAndSetProject(projectId);
    }
  }, [projectId]);

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
          if (value && value.workflow_id && value.workflow_id !== workflowId) {
            setWorkflowId(value.workflow_id);
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
      setLatestState(finalState);
      setIsLoading(false);
      return finalState;
    },
    [workflowId, cancelStepStream]
  );

  async function runWorkflow() {
    setIsLoading(true);
    setStepOutput(null);
    setMappedExportResult(null);
    if (newVideoHash === '') {
      return;
    }
    const runData: RunInput = {
      user_input: userMessage || '',
      streaming: true,
      ignore_running_workflows: true,
      user_email: userParams.user_email,
      video_hash: newVideoHash,
      length_seconds: lengthSeconds,
      timeline_name: projectName,
      structured_user_input: { video_type: videoType },
    };
    try {
      await run(userParams.workflow_id, runData, async (reader) => {
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
    setNewVideoHash(hash);
    setNewVideoFilename(filename);
    setWorkflowId('');
  };

  const restart = () => {
    setWorkflowId('');
    setStepOutput(null);
    setMappedExportResult(null);
    setBackendMessage('');
    setChatMessages([]);
    setProjectName('');
    setLengthSeconds(120);
    setVideoType(videoTypeExamples[0]);
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
                  setWorkflowId('');
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
                    setWorkflowId('');
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
                  setWorkflowId('');
                  setProjectName(e.target.value);
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
                  />
                  <ExportStepMenu userParams={userParams} stepName={stepName} />
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
