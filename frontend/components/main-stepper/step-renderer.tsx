import { ReactNode } from 'react';

import Chat from '@/components/chat/chat';
import StepOutput from '@/components/main-stepper/step-output';
import StepStreamingOutput from '@/components/main-stepper/step-streaming-output';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardFooter } from '@/components/ui/card';
import { Heading } from '@/components/ui/heading';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { ExportableStepWrapper, FrontendStepOutput } from '@/gen/openapi';

interface StepRendererProps {
  step: ExportableStepWrapper;
  stepOutput: FrontendStepOutput | null;
  outputText: string;
  stepInputPrompt: string;
  footer?: ReactNode;
  isNewStep: boolean;
  onRetry: (
    stepIndex: number,
    userMessage: string,
    callback: (aiMessage: string) => void
  ) => Promise<void>;
  stepIndex: number;
  isLoading: boolean;
  isInitialized: boolean;
  onCancelStep?: () => void;
}

function StepRenderer({
  step,
  stepOutput,
  outputText,
  stepInputPrompt,
  footer,
  onSubmit,
  stepIndex,
  isNewStep,
  isLoading,
  isInitialized,
  onCancelStep,
}: StepRendererProps) {
  const chatInitialMessages = stepInputPrompt
    ? [{ sender: 'AI', text: stepInputPrompt }]
    : [];
  if (stepOutput?.full_conversation) {
    stepOutput.full_conversation.forEach((msg) => {
      chatInitialMessages.push({ sender: msg.role, text: msg.value });
    });
  }

  const outputTextDefaultOpen =
    stepOutput === null || !stepOutput.step_outputs?.length;

  const onOutputFormSubmit = (data) => {
    // TODO should have a single submit button instead of two
    // and send chat message here
    console.log('onOutputFormSubmit', 'stepIndex', stepIndex, 'data', data);
    onSubmit({ stepIndex, userMesage: '', structuredUserInput: data });
  };
  return (
    <Card className="max-w-full shadow-none">
      <CardContent className="flex max-w-full p-0">
        {isLoading && (
          <div className="absolute top-0 left-0 w-full h-full bg-background/90 flex justify-center items-center flex-col gap-3 text-sm">
            {isInitialized ? 'Running step...' : 'Initializing...'}
            <LoadingSpinner size="large" />
            {onCancelStep && (
              <Button variant="secondary" onClick={onCancelStep}>
                Cancel
              </Button>
            )}
          </div>
        )}
        <div className="w-1/2 p-4">
          <Heading className="mb-3" size="sm">
            Chat
          </Heading>
          <Chat
            isNewStep={isNewStep}
            onNewMessage={(userMessage, callback) =>
              onSubmit({ stepIndex, userMessage, callback })
            }
            onEmptySubmit={(callback) => onSubmit({ stepIndex, callback })}
            initialMessages={chatInitialMessages}
          />
        </div>
        <div className="w-1/2 border-l p-4">
          <Heading className="mb-3" size="sm">
            Outputs
          </Heading>
          <StepStreamingOutput
            defaultOpen={outputTextDefaultOpen}
            value={outputText}
            step={step}
          />
          <StepOutput output={stepOutput} onSubmit={onOutputFormSubmit} />
        </div>
      </CardContent>
      {footer && <CardFooter>{footer}</CardFooter>}
    </Card>
  );
}

export default StepRenderer;
