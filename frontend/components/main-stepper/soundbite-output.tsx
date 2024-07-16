import { ArrowUpIcon } from '@radix-ui/react-icons';
import React, { FC } from 'react';
import { z } from 'zod';

import { Button } from '@/components/ui/button';
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

export const SoundbiteOutput: FC<OutputComponentProps> = ({
  value,
  exportResult,
  onSubmit,
  isLoading,
  form,
  allowModification,
}) => {
  const soundbiteTranscripts = value;
  const soundbiteClips = exportResult?.soundbites_videos || [];
  const onSubmitWrapper = (data: z.infer<typeof StructuredInputFormSchema>) => {
    if (!allowModification) {
      return;
    }
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
                        {allowModification ? (
                          <Switch
                            id={`identify_key_soundbites.soundbite_selection.${segmentIndex}`}
                            checked={field.value}
                            onCheckedChange={field.onChange}
                          />
                        ) : null}
                      </FormControl>
                      <FormMessage />
                      {allowModification ? (
                        <Label
                          htmlFor={`identify_key_soundbites.soundbite_selection.${segmentIndex}`}
                        >
                          {field.value ? 'Keep' : 'Remove'}
                        </Label>
                      ) : null}

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
          {allowModification ? (
            <Button disabled={isLoading} size="sm" type="submit">
              <ArrowUpIcon className="mr-2" />
              Update soundbites
            </Button>
          ) : null}
        </form>
      </Form>
    </div>
  );
};
