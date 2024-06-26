import React, { FC, useEffect, useRef, useState } from 'react';
import { z} from 'zod';
import { Button } from '@/components/ui/button';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Switch } from "@/components/ui/switch"

import {

  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from "@/components/ui/card"
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/ui/carousel"

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Label } from '@/components/ui/label';
import { FrontendStepOutput, StructuredUserInputInput } from '@/gen/openapi';
import { remoteVideoStreamURLForPath , getFunctionCallResults } from '@/lib/api';

const POLL_INTERVAL = 5000;

const removeEmptyVals = (d)=> {
  return Object.keys(d).reduce((acc, key) => {
    if (d[key].length) {
      acc[key] = [key];
    }
    return acc;
  }, {});
}

interface StepOutputProps {
  output: FrontendStepOutput;
  exportResult: Record<string, any> | null;
  exportCallId: string | null;
  onSubmit: (formData: StructuredUserInputInput) => void;
}

interface StepOutputItemProps {
  label: string;
  output: any;
  index: number;
  exportResult: Record<string, any>;
  onSubmit: (formData: StructuredUserInputInput) => void;
}

interface OutputComponentProps {
  value: any;
  exportResult: any;
  onSubmit: (formData: StructuredUserInputInput) => void;
}
const TranscriptOutput: FC<OutputComponentProps> = ({ value }) => {
  return <div className="mt-1">Transcript: {value}</div>;
};
const SoundbitesStateOutput: FC<OutputComponentProps> = ({ value }) => {
  return <div className="mt-1">Transcript chunks: {value.chunks.length}</div>;
};


const VideoStreamComponent: FC<{videoPath: string}> = ({videoPath}) => {
  const remoteUrl = remoteVideoStreamURLForPath(videoPath);
  return (
    <video width="320" height="240" controls>
      <source src={remoteUrl} type="video/mp4" />
      <track kind="captions" />
      Your browser does not support the video tag.
    </video>
  )
};
const SpeakerTaggingExamples: FC<{videoPaths: string[]}> = ({videoPaths}) => {
  return (
    <Carousel className="w-full max-w-xs">
      <CarouselContent>
        {videoPaths.map((videoPath, index) => (
          <CarouselItem key={index}>
            <div className="p-1">
              <Card>
                <CardContent className="flex aspect-square items-center justify-center p-6">
                  <VideoStreamComponent key={index} videoPath={videoPath} />
                </CardContent>
              </Card>
            </div>
          </CarouselItem>
        ))}
      </CarouselContent>
      <CarouselPrevious />
      <CarouselNext />
    </Carousel>
  )
}
export const OnScreenSpeakerIdentificationFormSchema = z.object({
  speaker_name_mapping: z.record(z.string()),
  speaker_tag_mapping: z.record(z.string(), z.boolean())
});


const OnScreenSpeakerIdentificationOutput: FC<OutputComponentProps> = (
  { value, exportResult, onSubmit }
) => {
  const speakerTaggingClips = exportResult.speaker_tagging_clips || {};
  const defaultNameMapping = Object.keys(speakerTaggingClips).reduce((acc, key) => {
    acc[key] = '';
    return acc;
  }, {});
  const defaultTagMapping = Object.keys(speakerTaggingClips).reduce((acc, key) => {
    acc[key] = value.includes(key.toLowerCase());
    return acc;
  }, {});
  const form = useForm<z.infer<typeof OnScreenSpeakerIdentificationFormSchema>>({
    resolver: zodResolver(OnScreenSpeakerIdentificationFormSchema),
    defaultValues: {
      speaker_name_mapping: defaultNameMapping,
      speaker_tag_mapping: defaultTagMapping
    },
  });
  const onSubmitWrapper = (data: z.infer<typeof OnScreenSpeakerIdentificationFormSchema>) => {
    data.speaker_name_mapping = removeEmptyVals(data.speaker_name_mapping);
    data.speaker_tag_mapping = removeEmptyVals(data.speaker_tag_mapping);
    onSubmit({remove_off_screen_speakers: data} as StructuredUserInputInput);
  };

  return (
    <div className="relative p-3">
    <Form {...form}>
      <form
        onSubmit={form.handleSubmit((data) => onSubmit(data))}
        className="w-2/3 space-y-6"
      >
        {Object.keys(speakerTaggingClips).map((speaker, index) => (
          <div key={index}>
          <FormLabel>Speaker {speaker} tagged as</FormLabel>
          <FormField
            control={form.control}
            name={`speaker_tag_mapping.${speaker}`}
            render={
              ({field})=>{
                return (
                  <FormItem>
                    <FormControl>
                      <Switch
                        id={`on-screen-switch-${speaker}`}
                        checked={field.value}
                        onCheckedChange={field.onChange}
                      />
                    </FormControl>
                    <FormMessage />
                    <Label htmlFor={`on-screen-switch-${speaker}`}>{(field.value) ? "On Screen" : "Off Screen"}</Label>
                  </FormItem>
                );
            }}
          />
          <FormField
            control={form.control}
            name={`speaker_name_mapping.${speaker}`}
            render={
              ({field})=>{
                return (
                  <FormItem>
                    <FormControl>
                      <React.Fragment>
                        <Input {...field} placeholder={`Enter new name for ${speaker}`}/>
                        <SpeakerTaggingExamples videoPaths={speakerTaggingClips[speaker]}/>
                      </React.Fragment>
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                );
            }}
          />
        </div>
        ))}
        <Button size="sm" disabled={form.formState.isSubmitting} type="submit" variant="secondary">
           Submit
        </Button>
      </form>
    </Form>
    </div>
  );
};

const outputComponentMapping = {
  current_transcript_text: TranscriptOutput,
  soundbites_state: SoundbitesStateOutput,
  on_screen_speakers: OnScreenSpeakerIdentificationOutput,
};

const StepOutputItem: FC<StepOutputItemProps> = ({ label, output, index, exportResult, onSubmit}) => {
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
        <Component value={output} exportResult={exportResult} onSubmit={onSubmit}/>
      </AccordionContent>
    </AccordionItem>
  );
};

const StepOutput: FC<StepOutputProps> = ({ output, onSubmit }) => {
  if (!output) {
    return <div className="text-muted-foreground">No outputs</div>;
  }
  const [exportResult, setExportResult] = useState<Record<string, any> | null>(output.export_result);
  const exportResultDone = useRef<bool>(output.export_result && Object.keys(output.export_result).length > 0);
  const timeoutId = useRef<NodeJS.Timeout | null>(null);
  const isComponentMounted = useRef<boolean>(true);
  const isPolling = useRef<boolean>(false);

  useEffect(() => {
    async function checkAndSetExportResultStatus() {
      if (isPolling.current) return; // Ensure only one polling request in flight
      isPolling.current = true;
      const statuses = await getFunctionCallResults([output.export_call_id]);
      if (statuses[0] && statuses[0].status === 'done') {
        setExportResult(statuses[0].output || null)
        exportResultDone.current = true;
      }
      isPolling.current = false;
    }

    const pollForStatuses = async () => {
      await checkAndSetExportResultStatus();
      if (isComponentMounted.current) {
        timeoutId.current = setTimeout(pollForStatuses, POLL_INTERVAL);
      }
    };

    if (output.export_call_id && !exportResultDone.current) {
      pollForStatuses();
    };
    return () => {
      if (timeoutId.current) {
        clearTimeout(timeoutId.current);
      }
    };

  }, [output.export_call_id]);

  return (
    <Accordion
      type="multiple"
      collapsible="true"
      className="mx-auto space-y-6 max-w-full"
    >
      {(exportResult === null && exportResult === undefined && Object.keys(exportResult).length === 0) ?
        <LoadingSpinner>Computing streamable/downloadable results</LoadingSpinner>
        : null
      }
      {Object.keys(output.step_outputs).map((key, index) => (
        <StepOutputItem
          key={index}
          label={key}
          output={output.step_outputs[key]}
          index={index}
          exportResult={exportResult}
          onSubmit={onSubmit}
        />
      ))}
    </Accordion>
  );
};

export default StepOutput;
