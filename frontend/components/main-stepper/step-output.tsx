import { FC } from 'react';

import StepStatusBadge from '@/components/main-stepper/step-status-badge';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import CodeBlock from '@/components/ui/code-block';
import { Label } from '@/components/ui/label';
import { CutTranscriptLinearWorkflowStepOutput } from '@/gen/openapi';
import { cn } from '@/lib/utils';

interface StepOutputProps {
  outputs: CutTranscriptLinearWorkflowStepOutput[];
}

const StepOutputItem: FC<{
  output: CutTranscriptLinearWorkflowStepOutput;
  index: number;
}> = ({ output, index }) => (
  <AccordionItem
    key={index}
    value={`item-${index}`}
    className="space-y-4 max-w-2xl"
  >
    <AccordionTrigger className="max-w-full gap-2 hover:no-underline">
      <div className="truncate font-semibold mr-auto">
        {output.step_name}.{output.substep_name}
      </div>
      <StepStatusBadge done={!!output.done} />
    </AccordionTrigger>
    <AccordionContent className="space-y-4">
      {output.user_feedback_request && (
        <div>
          <Label>User Feedback Request</Label>
          <div className="mt-1">{output.user_feedback_request}</div>
        </div>
      )}
      {output.partial_user_feedback_request && (
        <div>
          <Label>Partial User Feedback Request</Label>
          <div className="mt-1">{output.partial_user_feedback_request}</div>
        </div>
      )}
      {output.error && (
        <div className={cn('text-error')}>
          <Label>Error</Label>
          <div className="mt-1">{output.error}</div>
        </div>
      )}
      {output.step_inputs && (
        <div>
          <Label>Step Inputs</Label>
          <div className="mt-1">
            <CodeBlock code={JSON.stringify(output.step_inputs, null, 2)} />
          </div>
        </div>
      )}
      {output.step_outputs && (
        <div>
          <Label>Step Outputs</Label>
          <div className="mt-1">
            <CodeBlock code={JSON.stringify(output.step_outputs, null, 2)} />
          </div>
        </div>
      )}
      {output.export_result && (
        <div>
          <Label>Export Result</Label>
          <div className="mt-1">
            <CodeBlock code={JSON.stringify(output.export_result, null, 2)} />
          </div>
        </div>
      )}
      {output.export_call_id && (
        <div>
          <Label>Export Call ID</Label>
          <div className="mt-1">{output.export_call_id}</div>
        </div>
      )}
    </AccordionContent>
  </AccordionItem>
);

const StepOutput: FC<StepOutputProps> = ({ outputs }) => {
  if (!outputs || outputs.length === 0) {
    return <div className="text-muted-foreground">No outputs</div>;
  }

  return (
    <Accordion
      type="single"
      collapsible
      className="mx-auto space-y-6 max-w-full"
    >
      {outputs.map((output, index) => (
        <StepOutputItem key={index} output={output} index={index} />
      ))}
    </Accordion>
  );
};

export default StepOutput;
