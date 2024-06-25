'use client';

import { zodResolver } from '@hookform/resolvers/zod';
import { ChangeEvent, useState } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';

import { Button } from '@/components/ui/button';
import ExportStepMenu from '@/components/ui/export-step-menu';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { Textarea } from '@/components/ui/textarea';
import { useStepperForm } from '@/contexts/stepper-form-context';
import { ExportableStepWrapper } from '@/gen/openapi/api';

export const FormSchema = z.object({
  feedback: z.optional(z.string()),
});

interface StepperFormProps {
  systemPrompt: string;
  isInitialized: boolean;
  isLoading: boolean;
  stepIndex: number;
  userParams: any;
  step: ExportableStepWrapper;
  prompt: string;
  onRetry: (stepIndex: number, data: z.infer<typeof FormSchema>) => void;
  onSubmit: (stepIndex: number) => void;
  onCancelStep?: () => void;
}
export function StepperForm({
  systemPrompt,
  isInitialized,
  isLoading,
  stepIndex,
  userParams,
  step,
  prompt,
  onRetry,
  onSubmit,
  onCancelStep,
}: StepperFormProps) {
  // console.log('systemPrompt:', systemPrompt);
  // console.log('isLoading:', isLoading);
  // console.log('stepIndex:', stepIndex);
  // console.log('userParams:', userParams);
  // console.log('step:', step);
  // console.log('prompt:', prompt);
  // console.log('onRetry:', onRetry);
  // console.log('onCancelStep:', onCancelStep);

  const form = useForm<z.infer<typeof FormSchema>>({
    resolver: zodResolver(FormSchema),
  });
  const [textAreaValue, setTextAreaValue] = useState<string>('');
  const [prevUserMessage, setPrevUserMessage] = useState<string>('');

  const onSubmitInternal = (
    retry: boolean,
    data: z.infer<typeof FormSchema> | undefined
  ) => {
    setPrevUserMessage(textAreaValue);
    setTextAreaValue('');
    if (retry) {
      if (!data) {
        console.error('form data must be provided for retry');
      } else {
        onRetry(stepIndex, data);
      }
    } else {
      onSubmit(stepIndex + 1);
    }
  };

  const onRetryClick = () => {
    const formData = form.getValues();
    onSubmitInternal(true, formData);
  };

  const handleTextAreaChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    setTextAreaValue(event.target.value);
  };

  const { handleFormValueChange } = useStepperForm();

  return (
    <div className="relative p-3">
      <Form {...form}>
        <p>{systemPrompt}</p>
        <form
          onSubmit={form.handleSubmit((data) => onSubmitInternal(false, data))}
          className="w-2/3 space-y-6"
        >
          <FormField
            control={form.control}
            name="feedback"
            render={({ field }) => {
              const originalOnChange = field.onChange;
              field.onChange = (event) => {
                handleTextAreaChange(event);

                handleFormValueChange(form.getValues());
                originalOnChange(event);
              };
              field.value = textAreaValue;
              return (
                <FormItem>
                  <FormLabel>{prompt || ''}</FormLabel>
                  <FormControl>
                    <Textarea
                      placeholder="You can write anything you want."
                      className="resize-none"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              );
            }}
          />
          {prevUserMessage && (
            <p>
              <b>Previous message:</b>
              {prevUserMessage}
            </p>
          )}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {/*
              <Button
                size="sm"
                disabled={isLoading}
                type="button"
                onClick={onRetryClick}
                variant="secondary"
              >
                <ReloadIcon className="mr-2" />
                Retry
              </Button>
              <Button
                size="sm"
                disabled={isLoading}
                type="submit"
                variant="secondary"
              >
                <ReloadIcon className="mr-2" />
                Submit
              </Button>
                */}
              <ExportStepMenu
                userParams={userParams}
                stepName={step.name}
                substepName={step.name}
              />
            </div>
          </div>
        </form>
      </Form>
      {isLoading && (
        <div className="absolute top-0 left-0 w-full h-full bg-background/90 flex justify-center items-center flex-col gap-3 text-sm">
          {isInitialized ? 'Running step...' : 'Initializing...'}
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
