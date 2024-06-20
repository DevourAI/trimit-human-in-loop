'use client';

import { zodResolver } from '@hookform/resolvers/zod';
import { ReloadIcon } from '@radix-ui/react-icons';
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
  backendMessage: string;
  isLoading: boolean;
  stepIndex: number;
  userParams: any;
  step: ExportableStepWrapper;
  prompt: string;
  onRetry: (stepIndex: number, data: z.infer<typeof FormSchema>) => void;
  onCancelStep?: () => void;
}

export function StepperForm({
  systemPrompt,
  isLoading,
  stepIndex,
  userParams,
  step,
  prompt,
  onRetry,
  onCancelStep,
}: StepperFormProps) {
  const form = useForm<z.infer<typeof FormSchema>>({
    resolver: zodResolver(FormSchema),
  });
  const [textAreaValue, setTextAreaValue] = useState<string>('');
  const [prevUserMessage, setPrevUserMessage] = useState<string>('');

  const onSubmit = (data: z.infer<typeof FormSchema>) => {
    setPrevUserMessage(textAreaValue);
    setTextAreaValue('');
    onRetry(stepIndex, data);
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
          onSubmit={form.handleSubmit((data) => onSubmit(data))}
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
              <Button
                size="sm"
                disabled={isLoading}
                type="submit"
                variant="secondary"
              >
                <ReloadIcon className="mr-2" />
                Retry
              </Button>
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
