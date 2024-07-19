'use client';
import React, { useCallback, useEffect, useMemo, useState } from 'react';

import Chat from '@/components/chat/chat';
import { Message } from '@/components/chat/chat';
import StepOutput from '@/components/main-stepper/step-output';
import { Card, CardContent, CardFooter } from '@/components/ui/card';
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

export default function OneButtonGenerate({
  projectId,
}: {
  projectId: string;
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
  const [newVideoFilename, setNewVideoFilename] = useState<string>('');
  const [lengthSeconds, setLengthSeconds] = useState<number | null>(120);
  const [videoType, setVideoType] = useState<string | null>(
    'customer testimonial'
  );

  const [latestState, setLatestState] = useState<FrontendWorkflowState | null>(
    null
  );

  const [project, setProject] = useState<FrontendWorkflowProjection | null>(
    null
  );
  const [chatMessages, setChatMessages] = useState<Message[]>([]);

  const userParams = useMemo(
    () => ({
      user_email: userData.email,
      workflow_id: project?.id || null,
      video_hash: project?.video_hash || null,
    }),
    [userData.email, project]
  );
  useEffect(() => {
    async function fetchLatestState() {
      if (userParams.workflow_id === null) return;
      const data = await getLatestState(userParams.workflow_id);
      if (!data || Object.keys(data).length === 0) return;
      if (data.outputs && data.outputs.length) {
        // TODO do a check here that last step is "end"
        setStepOutput(data.outputs[data.outputs.length - 2]);
        setMappedExportResult(
          data.mapped_export_result[data.mapped_export_result.length - 2]
        );
      }
      setLatestState(data);
    }
    fetchLatestState();
  }, [userData, userParams, project?.video_hash]);

  const setAIMessageCallback = (aiMessage: string) => {
    setChatMessages((prevMessages) => [
      ...prevMessages,
      { sender: 'AI', text: aiMessage },
    ]);
  };

  useEffect(() => {
    async function fetchAndSetProject() {
      const project = await getWorkflowDetails(projectId);
      setProject(project);
    }
    if (projectId) {
      fetchAndSetProject();
    }
  }, [projectId]);

  const handleStepStream = useCallback(
    async (reader: ReadableStreamDefaultReader) => {
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
          }
        }
      );
      setLatestState(finalState);
      setStepOutput(
        finalState?.outputs?.length
          ? finalState.outputs[finalState.outputs.length - 2]
          : null
      );
      setMappedExportResult(
        finalState?.mapped_export_result?.length
          ? finalState.mapped_export_result[
              finalState.mapped_export_result.length - 2
            ]
          : null
      );

      setIsLoading(false);
      console.log('final state', finalState);
      return finalState;
    },
    []
  );

  async function runWorkflow() {
    setIsLoading(true);
    setStepOutput(null);
    setMappedExportResult(null);
    const runData: RunInput = {
      user_input: userMessage || '',
      streaming: true,
      ignore_running_workflows: true,
      user_email: userParams.user_email,
      video_hash: newVideoHash,
      length_seconds: lengthSeconds,
      structured_user_input: { video_type: videoType },
    };
    console.log('runData', runData, 'userParams', userParams);
    if (userParams.workflow_id === null) {
      return;
    }
    try {
      //await run(userParams.workflow_id, runData, async (reader) => {
      await run('', runData, async (reader) => {
        console.log('streaming callback');
        const finalState = await handleStepStream(reader);
        console.log('final state in streaming callback', finalState);
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
  const stepName =
    latestState?.all_steps[latestState.all_steps.length - 1].name || '';

  const handleVideoSelected = (hash: string, filename: string) => {
    setNewVideoHash(hash);
    setNewVideoFilename(filename);
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
        <Label htmlFor="lengthSeconds">Desired length of video (seconds)</Label>
        <Input
          id="lengthSeconds"
          value={lengthSeconds || ''}
          onChange={(e) =>
            setLengthSeconds(e.target.value ? parseFloat(e.target.value) : null)
          }
        />
        <Label htmlFor="videoType">Video type</Label>
        <Input
          id="videoType"
          value={videoType || ''}
          onChange={(e) => setVideoType(e.target.value)}
        />
        <div className="flex gap-3 w-full justify-between mb-3 items-center">
          Video: {newVideoFilename || 'None selected'}
        </div>

        <Card className="max-w-full shadow-none">
          <CardContent className="flex max-w-full p-0">
            <div className="w-1/2 p-4">
              <Heading className="mb-3" size="sm">
                Chat
              </Heading>
              <Chat
                onSubmit={runWorkflow}
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
