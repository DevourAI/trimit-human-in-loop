'use client';

import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
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
import { Input } from '@/components/ui/input';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { VideoFormSelector } from '@/components/ui/video-form-selector';
import { UploadedVideo } from '@/gen/openapi/api';

const DEFAULT_LENGTH_SECONDS = 120;
const DEFAULT_N_STAGES = 2;
export const WorkflowCreationFormSchema = z.object({
  project_name: z
    .string()
    .min(2, {
      message:
        'Project name must be at least 2 characters (or left blank to create one automatically).',
    })
    .optional()
    .or(z.literal('')),
  timeline_name: z
    .string()
    .min(2, {
      message:
        'Timeline name must be at least 2 characters (or left blank to create one automatically).',
    })
    .optional()
    .or(z.literal('')),
  video_type: z.string(),
  length_seconds: z.preprocess(
    (val) => parseInt(val as string, 10),
    z.number().default(DEFAULT_LENGTH_SECONDS)
  ),
  nstages: z.number().min(1).max(2).default(DEFAULT_N_STAGES),
  video_hash: z.string(),
});

interface WorkflowCreationFormProps {
  isLoading: boolean;
  userEmail: string;
  availableVideos: UploadedVideo[];
  onSubmit: (data: z.infer<typeof WorkflowCreationFormSchema>) => void;
  onCancelStep?: () => void;
}
export function WorkflowCreationForm({
  isLoading,
  availableVideos,
  onSubmit,
  onCancelStep,
}: WorkflowCreationFormProps) {
  const form = useForm<z.infer<typeof WorkflowCreationFormSchema>>({
    resolver: zodResolver(WorkflowCreationFormSchema),
    defaultValues: {
      project_name: '',
      timeline_name: '',
      length_seconds: DEFAULT_LENGTH_SECONDS,
      nstages: DEFAULT_N_STAGES,
      video_hash:
        availableVideos !== undefined &&
        availableVideos !== null &&
        availableVideos.length
          ? availableVideos[0].video_hash
          : '',
    },
  });

  return (
    <div className="relative p-3">
      <Form {...form}>
        <form
          onSubmit={form.handleSubmit((data) => onSubmit(data))}
          className="w-2/3 space-y-6"
        >
          <FormField
            control={form.control}
            name="project_name"
            render={({ field }) => {
              return (
                <FormItem>
                  <FormLabel>Project name (optional)</FormLabel>
                  <FormControl>
                    <Input {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              );
            }}
          />
          <FormField
            control={form.control}
            name="timeline_name"
            render={({ field }) => {
              return (
                <FormItem>
                  <FormLabel>Timeline name (optional)</FormLabel>
                  <FormControl>
                    <Input {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              );
            }}
          />
          <FormField
            control={form.control}
            name="video_hash"
            render={({ field }) => {
              return (
                <VideoFormSelector
                  formLabel="Video"
                  onChange={field.onChange}
                  defaultValue={field.value}
                  availableVideos={availableVideos}
                />
              );
            }}
          />
          <FormField
            control={form.control}
            name="length_seconds"
            render={({ field }) => {
              return (
                <FormItem>
                  <FormLabel>Desired length of video (seconds)</FormLabel>
                  <FormControl>
                    <Input {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              );
            }}
          />
          <FormField
            control={form.control}
            name="video_type"
            render={({ field }) => {
              return (
                <FormItem>
                  <FormLabel>Video type</FormLabel>
                  <FormControl>
                    <Input {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              );
            }}
          />
          <FormField
            control={form.control}
            name="nstages"
            render={({ field }) => {
              return (
                <FormItem>
                  <FormLabel>
                    Use one or two stages to optionally first cut the video down
                    to a longer length.
                  </FormLabel>
                  <FormControl>
                    <RadioGroup
                      onValueChange={field.onChange}
                      defaultValue={`${DEFAULT_N_STAGES}`}
                      className="flex flex-col space-y-1"
                    >
                      <FormItem className="flex items-center space-x-3 space-y-0">
                        <FormControl>
                          <RadioGroupItem value="1" />
                        </FormControl>
                        <FormLabel className="font-normal">1 Stage</FormLabel>
                      </FormItem>
                      <FormItem className="flex items-center space-x-3 space-y-0">
                        <FormControl>
                          <RadioGroupItem value="2" />
                        </FormControl>
                        <FormLabel className="font-normal">2 Stages</FormLabel>
                      </FormItem>
                    </RadioGroup>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              );
            }}
          />
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Button
                size="sm"
                disabled={isLoading}
                type="submit"
                variant="secondary"
              >
                Submit
              </Button>
            </div>
          </div>
        </form>
      </Form>
      {isLoading && (
        <div className="absolute top-0 left-0 w-full h-full bg-background/90 flex justify-center items-center flex-col gap-3 text-sm">
          Running step...
          <LoadingSpinner size="large" />
          {onCancelStep && (
            <Button variant="secondary" onClick={onCancelStep}>
              Cancel
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
