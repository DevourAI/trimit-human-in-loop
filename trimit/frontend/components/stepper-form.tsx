"use client"

import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import { z } from "zod"

import { Button } from "@/components/ui/button"
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

const FormSchema = z.object({
  feedback: z
    .string()
    // .min(10, {
      // message: "Bio must be at least 10 characters.",
    // })
    // .max(160, {
      // message: "Bio must not be longer than 30 characters.",
  //}),
})

export function StepperForm({ stepIndex, userData, step, onSubmit }) {
  const form = useForm<z.infer<typeof FormSchema>>({
    resolver: zodResolver(FormSchema),
  })

  function innerOnSubmit(data: z.infer<typeof FormSchema>) {
    onSubmit(stepIndex, data)
    toast({
      title: "You submitted the following values:",
      description: (
        <pre className="mt-2 w-[340px] rounded-md bg-slate-950 p-4">
          <code className="text-white">{JSON.stringify(data, null, 2)}</code>
        </pre>
      ),
    })
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(innerOnSubmit)} className="w-2/3 space-y-6">
        <FormField
          control={form.control}
          name="feedback"
          render={({ field }) => (
            <FormItem>
              <FormLabel>{step.name}</FormLabel>
              <FormControl>
                <Textarea
                  placeholder={step.input?.prompt || "What do you want to create?"}
                  className="resize-none"
                  {...field}
                />
              </FormControl>
              <FormDescription>
                You can write anything you want.
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />
        <Button type="submit">Submit</Button>
      </form>
    </Form>
  )
}
