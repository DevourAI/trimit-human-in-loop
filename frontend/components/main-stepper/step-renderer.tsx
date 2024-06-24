import { ReactNode } from 'react';

import Chat from '@/components/chat/chat';
import StepOutput from '@/components/main-stepper/step-output';
import { Card, CardContent, CardFooter } from '@/components/ui/card';
import { Heading } from '@/components/ui/heading';
import { ExportableStepWrapper, FrontendStepOutput } from '@/gen/openapi';

interface StepRendererProps {
  step: ExportableStepWrapper;
  stepOutput: FrontendStepOutput | null;
  stepInputPrompt: string;
  footer?: ReactNode;
  onRetry: (
    stepIndex: number,
    userMessage: string,
    callback: (aiMessage: string) => void
  ) => Promise<void>;
  stepIndex: number;
}

function StepRenderer({
  step,
  stepOutput,
  stepInputPrompt,
  footer,
  onRetry,
  stepIndex,
}: StepRendererProps) {
  const chatInitialMessages = stepInputPrompt
    ? [{ sender: 'AI', text: stepInputPrompt }]
    : [];
  if (stepOutput?.full_conversation) {
    stepOutput.full_conversation.forEach((msg) => {
      chatInitialMessages.push({ sender: msg.role, text: msg.value });
    });
  }

  return (
    <Card className="max-w-full shadow-none">
      <CardContent className="flex max-w-full p-0">
        <div className="w-1/2 p-4">
          <Heading className="mb-3" size="sm">
            Chat
          </Heading>
          <Chat
            onNewMessage={(msg, callback) => onRetry(stepIndex, msg, callback)}
            initialMessages={chatInitialMessages}
          />
        </div>
        <div className="w-1/2 border-l p-4">
          <Heading className="mb-3" size="sm">
            Outputs
          </Heading>
          <StepOutput outputs={stepOutput ? [stepOutput] : []} />
        </div>
      </CardContent>
      {footer && <CardFooter>{footer}</CardFooter>}
    </Card>
  );
}

export default StepRenderer;
