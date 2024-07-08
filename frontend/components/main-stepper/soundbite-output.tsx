import { ArrowUpIcon } from '@radix-ui/react-icons';
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
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { VideoStream } from '@/components/ui/video-stream';
import { StructuredInputFormSchema } from '@/contexts/structured-input-form-context';
import { OutputComponentProps } from '@/lib/types';
import { removeEmptyVals } from '@/lib/utils';

const Transcript: FC<{ text: string }> = ({ text }) => {
  return <p>{text}</p>;
};

const SoundbiteExamples: FC<{
  videoPaths: string[];
  soundbiteTranscripts: string[];
}> = ({ videoPaths }) => {
  return (
    <Carousel className="w-full max-w-xs">
      <CarouselContent>
        {soundbiteTranscripts.map((soundbiteTranscript, index) => (
          <CarouselItem key={index}>
            <div className="p-1">
              <Card>
                <CardContent className="flex aspect-square items-center justify-center p-6">
                  <VideoStream
                    videoPath={
                      videoPaths && videoPaths.length > index
                        ? videoPaths[index]
                        : ''
                    }
                  />
                  <Transcript text={soundbiteTranscript} />
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

export const SoundbiteOutput: FC<OutputComponentProps> = ({
  value,
  exportResult,
  onSubmit,
  isLoading,
  form,
}) => {
  const soundbiteTranscripts = value;
  const soundbiteClips = exportResult?.soundbites_videos || [];
  const onSubmitWrapper = (data: z.infer<typeof StructuredInputFormSchema>) => {
    console.log('submitting structured form');
    data.identify_key_soundbites.soundbite_selection = removeEmptyVals(
      data.identify_key_soundbites.soundbite_selection
    );
    onSubmit();
  };

  //form.handleSubmit(onSubmitWrapper)}
  // TODO wrap all these form inputs in a carousel
  return (
    <div className="relative p-3">
      <Form {...form}>
        <form
          onSubmit={form.handleSubmit(onSubmitWrapper)}
          className="w-2/3 space-y-6"
        >
          {soundbiteTranscripts.map(([segmentIndex, text], index) => (
            <div key={index}>
              <FormLabel>Soundbite {index}</FormLabel>
              <FormField
                control={form.control}
                name={`identify_key_soundbites.soundbite_selection.${segmentIndex}`}
                render={({ field }) => {
                  return (
                    <FormItem>
                      <FormControl>
                        <Switch
                          id={`identify_key_soundbites.soundbite_selection.${segmentIndex}`}
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </FormControl>
                      <FormMessage />
                      <Label
                        htmlFor={`identify_key_soundbites.soundbite_selection.${segmentIndex}`}
                      >
                        {field.value ? 'Keep' : 'Remove'}
                      </Label>

                      <VideoStream
                        videoPath={
                          soundbiteClips && soundbiteClips.length > index
                            ? soundbiteClips[index]
                            : ''
                        }
                      />
                      <Transcript text={text} />
                    </FormItem>
                  );
                }}
              />
            </div>
          ))}
          <Button disabled={isLoading} size="sm" type="submit">
            <ArrowUpIcon className="mr-2" />
            Update soundbites
          </Button>
        </form>
      </Form>
    </div>
  );
};
