import { FC } from 'react';

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Label } from '@/components/ui/label';
import { FrontendStepOutput } from '@/gen/openapi';

interface StepOutputProps {
  output: FrontendStepOutput;
}

interface StepOutputItemProps {
  label: string;
  output: any;
  index: number;
}

const TranscriptOutput: FC<{ value: string }> = ({ value }) => {
  return <div className="mt-1">Transcript: {value}</div>;
};
const SoundbitesStateOutput: FC<{ value: any }> = ({ value }) => {
  return <div className="mt-1">Transcript chunks: {value.chunks.length}</div>;
};

const outputComponentMapping = {
  current_transcript_text: TranscriptOutput,
  soundbites_state: SoundbitesStateOutput,
};
const StepOutputItem: FC<StepOutputItemProps> = ({ label, output, index }) => {
  const Component = outputComponentMapping[label];
  if (Component === undefined) {
    return null;
  }
  return (
    <AccordionItem
      key={index}
      value={`item-${index}`}
      className="space-y-4 max-w-2xl"
    >
      <AccordionTrigger className="max-w-full gap-2 hover:no-underline">
        <div>
          <Label>{label}</Label>
        </div>
      </AccordionTrigger>
      <AccordionContent className="space-y-4">
        <Component value={output} />
      </AccordionContent>
    </AccordionItem>
  );
};

const StepOutput: FC<StepOutputProps> = ({ output }) => {
  if (!output) {
    return <div className="text-muted-foreground">No outputs</div>;
  }

  return (
    <Accordion
      type="multiple"
      collapsible
      className="mx-auto space-y-6 max-w-full"
    >
      {Object.keys(output.step_outputs).map((key, index) => (
        <StepOutputItem
          key={index}
          label={key}
          output={output.step_outputs[key]}
          index={index}
        />
      ))}
    </Accordion>
  );
};

export default StepOutput;
