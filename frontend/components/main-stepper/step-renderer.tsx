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
            onNewMessage={(msg, callback) => onSubmit(stepIndex, msg, callback)}
            onEmptySubmit={(callback) => onSubmit(stepIndex, '', callback)}
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
          <StepOutput output={stepOutput} />
        </div>
      </CardContent>
      {footer && <CardFooter>{footer}</CardFooter>}
    </Card>
  );
}

export default StepRenderer;
