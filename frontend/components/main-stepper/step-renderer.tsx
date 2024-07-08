import { ReactNode } from 'react';

import { Message } from '@/components/chat/chat';
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
  onSubmit: (options: any) => Promise<void>;
  stepIndex: number;
  isLoading: boolean;
  isInitialized: boolean;
  onCancelStep?: () => void;
  backendMessage: string;
  setUserMessage: (userMessage: string) => void;
  userMessage: string;
  chatMessages: Message[];
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
  backendMessage,
  userMessage,
  setUserMessage,
  chatMessages,
}: StepRendererProps) {
  const outputTextDefaultOpen =
    stepOutput === null || !stepOutput.step_outputs?.length;

  const onOutputFormSubmit = () => {
    // TODO should have a single submit button instead of two
    // and send chat message here
    onSubmit({ useStructuredInput: true });
  };
  return (
    <Card className="max-w-full shadow-none">
      <CardContent className="flex max-w-full p-0">
        <div className="w-1/2 p-4">
          <Heading className="mb-3" size="sm">
            Chat
          </Heading>
          <Chat
            isNewStep={isNewStep}
            onSubmit={onSubmit}
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
          <StepStreamingOutput
            defaultOpen={outputTextDefaultOpen}
            value={outputText}
            step={step}
          />
          <StepOutput
            isLoading={isLoading}
            output={stepOutput}
            onSubmit={onOutputFormSubmit}
          />
        </div>
      </CardContent>
      {footer && (
        <CardFooter>
          {footer}
          {isLoading && (
            <div className="absolute bottom-0 left-0 w-full bg-background/90 flex justify-center items-center flex-col gap-3 text-sm">
              {isInitialized
                ? `${backendMessage || 'Running step'}...`
                : 'Initializing...'}
              <LoadingSpinner size="large" />
              {onCancelStep && (
                <Button variant="secondary" onClick={onCancelStep}>
                  Cancel
                </Button>
              )}
            </div>
          )}
        </CardFooter>
      )}
    </Card>
  );
}

export default StepRenderer;
