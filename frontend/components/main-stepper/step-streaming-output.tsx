import { FC } from 'react';

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Label } from '@/components/ui/label';

interface StepStreamingOutputProps {
  defaultOpen: boolean;
  value: string;
}

const StepStreamingOutput: FC<StepStreamingOutputProps> = ({
  defaultOpen,
  value,
}) => {
  if (!value) {
    return null;
  }

  return (
    <Accordion
      type="single"
      collapsible
      className="mx-auto space-y-6 max-w-full"
      defaultValue={defaultOpen ? 'item-1' : null}
    >
      <AccordionItem className="space-y-4 max-w-2xl" value="item-1">
        <AccordionTrigger className="max-w-full gap-2 hover:no-underline">
          <div className="truncate font-semibold mr-auto">
            <Label>Streaming Output</Label>
          </div>
        </AccordionTrigger>
        <AccordionContent className="space-y-4">
          <div className="mt-1">
            <p>{value}</p>
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
};

export default StepStreamingOutput;
