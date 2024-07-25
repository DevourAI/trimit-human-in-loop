import React, { FC } from 'react';
import type { UseFormReturn } from 'react-hook-form';

import { OnScreenSpeakerIdentificationOutput } from '@/components/main-stepper/on-screen-speaker-identification-output';
import { SoundbiteOutput } from '@/components/main-stepper/soundbite-output';
import { VideoOutput } from '@/components/main-stepper/video-output';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Label } from '@/components/ui/label';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { useStructuredInputForm } from '@/contexts/structured-input-form-context';
import { FrontendStepOutput } from '@/gen/openapi';
import { OutputComponentProps } from '@/lib/types';

//const POLL_INTERVAL = 5000;
const OUTPUT_LABEL_MAP = {
  current_transcript_text: 'Current Video & Transcript',
  on_screen_speakers: 'On Screen Speakers',
  current_soundbites_iter_text: 'Key Selects',
};

interface StepOutputProps {
  output: FrontendStepOutput;
  exportResult: Record<string, any> | null;
  exportCallId: string | null;
  onSubmit: () => void;
  isLoading: boolean;
  allowModification: boolean;
}

interface StepOutputItemProps {
  name: string;
  label: string;
  output: any;
  index: number;
  exportResult: Record<string, any>;
  onSubmit: () => void;
  form: UseFormReturn;
  isLoading: boolean;
  allowModification: boolean;
}

const StoryOutput: FC<OutputComponentProps> = ({ value }) => {
  return <div className="mt-1">Story: {value}</div>;
};

const outputComponentMapping = {
  current_transcript_text: VideoOutput,
  current_soundbites_iter_text: SoundbiteOutput,
  on_screen_speakers: OnScreenSpeakerIdentificationOutput,
  story: StoryOutput,
};

const StepOutputItem: FC<StepOutputItemProps> = ({
  name,
  label,
  output,
  index,
  exportResult,
  onSubmit,
  form,
  isLoading,
  allowModification,
}) => {
  const Component = outputComponentMapping[name];
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
        <Component
          value={output}
          exportResult={exportResult}
          onSubmit={onSubmit}
          form={form}
          isLoading={isLoading}
          allowModification={allowModification}
        />
      </AccordionContent>
    </AccordionItem>
  );
};

const StepOutput: FC<StepOutputProps> = ({
  output,
  onSubmit,
  isLoading,
  allowModification,
}) => {
  // const [exportResult, setExportResult] = useState<Record<string, any> | null>(
  // output?.export_result || null
  // );
  // const exportResultDone = useRef<bool>(
  // output.export_result && Object.keys(output.export_result).length > 0
  // );
  // const timeoutId = useRef<NodeJS.Timeout | null>(null);
  // const isComponentMounted = useRef<boolean>(true);
  // const isPolling = useRef<boolean>(false);

  // useEffect(() => {
  // async function checkAndSetExportResultStatus() {
  // if (isPolling.current) return; // Ensure only one polling request in flight
  // isPolling.current = true;
  // const statuses = await getFunctionCallResults([output.export_call_id]);
  // if (statuses[0] && statuses[0].status === 'done') {
  // setExportResult(statuses[0].output || null);
  // exportResultDone.current = true;
  // }
  // isPolling.current = false;
  // }

  // const pollForStatuses = async () => {
  // await checkAndSetExportResultStatus();
  // if (isComponentMounted.current && !exportResultDone.current) {
  // timeoutId.current = setTimeout(pollForStatuses, POLL_INTERVAL);
  // }
  // };

  // if (output.export_call_id && !exportResultDone.current) {
  // pollForStatuses();
  // }
  // return () => {
  // if (timeoutId.current) {
  // clearTimeout(timeoutId.current);
  // }
  // };
  // }, [output.export_call_id]);

  const { form, exportResult } = useStructuredInputForm();
  console.log('exportResult from form', exportResult);
  if (!output) {
    return <div className="text-muted-foreground">No outputs</div>;
  }
  return (
    <Accordion
      type="multiple"
      collapsible="true"
      className="mx-auto space-y-6 max-w-full"
    >
      {exportResult === null &&
      exportResult === undefined &&
      Object.keys(exportResult).length === 0 ? (
        <LoadingSpinner>
          Computing streamable/downloadable results
        </LoadingSpinner>
      ) : null}
      {Object.keys(output.step_outputs || []).map((key, index) => (
        <StepOutputItem
          isLoading={isLoading}
          key={index}
          name={key}
          label={OUTPUT_LABEL_MAP[key] || key}
          output={output.step_outputs[key]}
          index={index}
          exportResult={exportResult}
          onSubmit={onSubmit}
          form={form}
          allowModification={allowModification}
        />
      ))}
    </Accordion>
  );
};

export default StepOutput;
