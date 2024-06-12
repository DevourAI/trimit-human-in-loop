"use client"

import { useState, useRef } from 'react';
import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import { z } from "zod"
import {
  useStepper,
} from "@/components/ui/stepper"

import { Button } from "@/components/ui/button"
import DownloadButtons from "@/components/ui/download-buttons"
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form"
import { Textarea } from "@/components/ui/textarea"
import { toast } from "@/components/ui/use-toast"
import Ansi from "ansi-to-react";



const FormSchema = z.object({
  feedback: z.optional(z.string())
})

const AnsiFormattedText = ({ text }) => {
  //return <Ansi>test{text}</Ansi>;
  return <p>{text}</p>;
};

export function StepperForm({
  systemPrompt,
  isLoading,
  undoLastStep,
  stepIndex,
  userParams,
  step,
  prompt,
  onSubmit
}) {
  const {
    activeStep
  } = useStepper()
  const form = useForm<z.infer<typeof FormSchema>>({
    resolver: zodResolver(FormSchema),
  })
  const [textAreaValue, setTextAreaValue] = useState('');
  const [prevUserMessage, setPrevUserMessage] = useState('');

  function innerOnSubmit(retry, data: z.infer<typeof FormSchema>) {
    setPrevUserMessage(textAreaValue)
    setTextAreaValue('');
    onSubmit(stepIndex, retry, data)
  }

  function onRetryClick() {
    const formData = form.getValues();
    innerOnSubmit(true, formData)
  }

  const handleTextAreaChange = (event) => {
    setTextAreaValue(event.target.value)
  }

  const textAreaRef = useRef<HTMLTextAreaElement>(null);

  return (
    <Form {...form}>
      <p><Ansi useClasses>{stepIndex == activeStep ? systemPrompt || '': ''}</Ansi></p>
      <form onSubmit={form.handleSubmit((data)=>innerOnSubmit(false, data))} className="w-2/3 space-y-6">
        <FormField
          control={form.control}
          name="feedback"
          render={({ field }) => {
            const originalOnChange = field.onChange
            field.onChange = (event) => {
              handleTextAreaChange(event)
              originalOnChange()
            }
            field.value = textAreaValue
            return (
            <FormItem>
              <FormLabel><Ansi useClasses>{prompt || ''}</Ansi></FormLabel>
              <FormControl>
                <Textarea
                  ref={textAreaRef}
                  placeholder="You can write anything you want."
                  className="resize-none"
                  {...field}
                />
              </FormControl>
            {/* <FormDescription>
                // You can write anything you want.
              </FormDescription> */}
              <FormMessage />
            </FormItem>
            )
          }}
        />
        <p><b>Previous message:</b>{prevUserMessage}</p>
        <Button disabled={isLoading} type="submit">Submit</Button>
        <Button disabled={isLoading} type="button" onClick={onRetryClick}>Retry</Button>
      </form>
      <DownloadButtons userParams={userParams} stepName={step.step_name} substepName={step.name} />
      <Button className="w-1/12" onClick={undoLastStep} >Undo</Button>
    </Form>
  )
}
