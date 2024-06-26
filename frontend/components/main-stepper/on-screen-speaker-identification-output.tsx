import { zodResolver } from '@hookform/resolvers/zod';
import React, { FC } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';

import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from '@/components/ui/carousel';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { VideoStream } from '@/components/ui/video-stream';
import { StructuredUserInputInput } from '@/gen/openapi';
import { OutputComponentProps } from '@/lib/types';

const removeEmptyVals = (d) => {
  return Object.keys(d).reduce((acc, key) => {
    if (d[key].length) {
      acc[key] = [key];
    }
    return acc;
  }, {});
};

const SpeakerTaggingExamples: FC<{ videoPaths: string[] }> = ({
  videoPaths,
}) => {
  return (
    <Carousel className="w-full max-w-xs">
      <CarouselContent>
        {videoPaths.map((videoPath, index) => (
          <CarouselItem key={index}>
            <div className="p-1">
              <Card>
                <CardContent className="flex aspect-square items-center justify-center p-6">
                  <VideoStream key={index} videoPath={videoPath} />
                </CardContent>
              </Card>
            </div>
          </CarouselItem>
        ))}
      </CarouselContent>
      <CarouselPrevious />
      <CarouselNext />
    </Carousel>
  );
};
export const OnScreenSpeakerIdentificationFormSchema = z.object({
  speaker_name_mapping: z.record(z.string()),
  speaker_tag_mapping: z.record(z.string(), z.boolean()),
});

export const OnScreenSpeakerIdentificationOutput: FC<OutputComponentProps> = ({
  value,
  exportResult,
  onSubmit,
}) => {
  const speakerTaggingClips = exportResult.speaker_tagging_clips || {};
  const defaultNameMapping = Object.keys(speakerTaggingClips).reduce(
    (acc, key) => {
      acc[key] = '';
      return acc;
    },
    {}
  );
  const defaultTagMapping = Object.keys(speakerTaggingClips).reduce(
    (acc, key) => {
      acc[key] = value.includes(key.toLowerCase());
      return acc;
    },
    {}
  );
  const form = useForm<z.infer<typeof OnScreenSpeakerIdentificationFormSchema>>(
    {
      resolver: zodResolver(OnScreenSpeakerIdentificationFormSchema),
      defaultValues: {
        speaker_name_mapping: defaultNameMapping,
        speaker_tag_mapping: defaultTagMapping,
      },
    }
  );
  const onSubmitWrapper = (
    data: z.infer<typeof OnScreenSpeakerIdentificationFormSchema>
  ) => {
    data.speaker_name_mapping = removeEmptyVals(data.speaker_name_mapping);
    data.speaker_tag_mapping = removeEmptyVals(data.speaker_tag_mapping);
    onSubmit({ remove_off_screen_speakers: data } as StructuredUserInputInput);
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
                render={({ field }) => {
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
                      <Label htmlFor={`on-screen-switch-${speaker}`}>
                        {field.value ? 'On Screen' : 'Off Screen'}
                      </Label>
                    </FormItem>
                  );
                }}
              />
              <FormField
                control={form.control}
                name={`speaker_name_mapping.${speaker}`}
                render={({ field }) => {
                  return (
                    <FormItem>
                      <FormControl>
                        <React.Fragment>
                          <Input
                            {...field}
                            placeholder={`Enter new name for ${speaker}`}
                          />
                          <SpeakerTaggingExamples
                            videoPaths={speakerTaggingClips[speaker]}
                          />
                        </React.Fragment>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  );
                }}
              />
            </div>
          ))}
          <Button
            size="sm"
            disabled={form.formState.isSubmitting}
            type="submit"
            variant="secondary"
          >
            Submit
          </Button>
        </form>
      </Form>
    </div>
  );
};
