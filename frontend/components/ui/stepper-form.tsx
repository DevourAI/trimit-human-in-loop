'use client';

import { useState, useRef, ChangeEvent } from 'react';
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm, SubmitHandler } from 'react-hook-form';
import { z } from 'zod';
import { useStepper } from '@/components/ui/stepper';

import { Button } from '@/components/ui/button';
import ExportStepMenu from '@/components/ui/export-step-menu';
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';

import { Textarea } from '@/components/ui/textarea';
import { toast } from '@/components/ui/use-toast';

export const FormSchema = z.object({
  feedback: z.optional(z.string()),
});

interface StepperFormProps {
  systemPrompt: string;
  isLoading: boolean;
  stepIndex: number;
  userParams: any;
  step: { step_name: string; name: string };
  prompt: string;
  onSubmit: (
    stepIndex: number,
    retry: boolean,
    data: z.infer<typeof FormSchema>
  ) => void;
}

export function StepperForm({
  systemPrompt,
  isLoading,
  stepIndex,
  userParams,
  step,
  prompt,
  onSubmit,
}: StepperFormProps) {
  const { activeStep } = useStepper();
  const form = useForm<z.infer<typeof FormSchema>>({
    resolver: zodResolver(FormSchema),
  });
  const [textAreaValue, setTextAreaValue] = useState<string>('');
  const [prevUserMessage, setPrevUserMessage] = useState<string>('');

  const innerOnSubmit: SubmitHandler<z.infer<typeof FormSchema>> = (
    data: z.infer<typeof FormSchema>,
    retry = false
  ) => {
    setPrevUserMessage(textAreaValue);
    setTextAreaValue('');
    onSubmit(stepIndex, retry, data);
  };

  const onRetryClick = () => {
    const formData = form.getValues();
    innerOnSubmit(formData, true);
  };

  const handleTextAreaChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    setTextAreaValue(event.target.value);
  };

  const textAreaRef = useRef<HTMLTextAreaElement>(null);

  return (
    <Form {...form}>
      <p>{stepIndex == activeStep ? systemPrompt || '' : ''}</p>
      <form
        onSubmit={form.handleSubmit((data) => innerOnSubmit(data, false))}
        className="w-2/3 space-y-6"
      >
        <FormField
          control={form.control}
          name="feedback"
          render={({ field }) => {
            const originalOnChange = field.onChange;
            field.onChange = (event) => {
              handleTextAreaChange(event);
              originalOnChange(event);
            };
            field.value = textAreaValue;
            return (
              <FormItem>
                <FormLabel>{prompt || ''}</FormLabel>
                <FormControl>
                  <Textarea
                    ref={textAreaRef}
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
        <Button disabled={isLoading} type="submit">
          Submit
        </Button>
        <Button disabled={isLoading} type="button" onClick={onRetryClick}>
          Retry
        </Button>
      </form>
      <ExportStepMenu
        userParams={userParams}
        stepName={step.step_name}
        substepName={step.name}
      />
    </Form>
  );
}
