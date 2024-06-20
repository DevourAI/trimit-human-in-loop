import { ReactNode } from 'react';

import Chat from '@/components/chat/chat';
import StepOutput from '@/components/main-stepper/step-output';
import { Card, CardContent, CardFooter } from '@/components/ui/card';
import { Heading } from '@/components/ui/heading';
import { ExportableStepWrapper } from '@/gen/openapi';

interface StepRendererProps {
  step: ExportableStepWrapper;
  footer?: ReactNode;
}

function StepRenderer({ step, footer }: StepRendererProps) {
  return (
    <Card className="max-w-full shadow-none">
      <CardContent className="flex max-w-full p-0">
        <div className="w-1/2 p-4">
          <Heading className="mb-3" size="sm">
            Chat
          </Heading>
          <Chat />
        </div>
        <div className="w-1/2 border-l p-4">
          <Heading className="mb-3" size="sm">
            Outputs
          </Heading>
          <StepOutput outputs={[]} />
        </div>
      </CardContent>
      {footer && <CardFooter>{footer}</CardFooter>}
    </Card>
  );
}

export default StepRenderer;
