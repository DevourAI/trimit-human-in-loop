import React, { FC } from 'react';
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
import { OutputComponentProps } from '@/lib/types';

const removeEmptyVals = (d) => {
  return Object.keys(d).reduce((acc, key) => {
    if (d[key] !== undefined && d[key] !== null && d[key] !== '') {
      acc[key] = d[key];
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

export const OnScreenSpeakerIdentificationOutput: FC<OutputComponentProps> = ({
  value,
  exportResult,
  onSubmit,
  form,
}) => {
  const speakerTaggingClips = exportResult?.speaker_tagging_clips || {};
  const onSubmitWrapper = (data: z.infer<typeof StructuredInputFormSchema>) => {
    data.remove_off_screen_speakers.speaker_name_mapping = removeEmptyVals(
      data.remove_off_screen_speakers.speaker_name_mapping
    );
    data.remove_off_screen_speakers.speaker_tag_mapping = removeEmptyVals(
      data.remove_off_screen_speakers.speaker_tag_mapping
    );
  };

  return (
    <div className="relative p-3">
      <Form {...form}>
        <form
          onSubmit={form.handleSubmit((data) => onSubmitWrapper(data))}
          className="w-2/3 space-y-6"
        >
          {Object.keys(speakerTaggingClips).map((speaker, index) => (
            <div key={index}>
              <FormLabel>Speaker {speaker} tagged as</FormLabel>
              <FormField
                control={form.control}
                name={`remove_off_screen_speakers.speaker_tag_mapping.${speaker}`}
                render={({ field }) => {
                  return (
                    <FormItem>
                      <FormControl>
                        <Switch
                          id={`remove_off_screen_speakers.speaker_tag_mapping.${speaker}`}
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </FormControl>
                      <FormMessage />
                      <Label
                        htmlFor={`remove_off_screen_speakers.speaker_tag_mapping.${speaker}`}
                      >
                        {field.value ? 'On Screen' : 'Off Screen'}
                      </Label>
                    </FormItem>
                  );
                }}
              />
              <FormField
                control={form.control}
                name={`remove_off_screen_speakers.speaker_name_mapping.${speaker}`}
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
