import React, {ReactNode, createContext, useContext, useState} from 'react';
import {z} from 'zod';

export const FormSchema = z.object({
  feedback: z.optional(z.string()),
});
interface FormContextProps {
  stepperFormValues: z.infer<typeof FormSchema>;
  handleFormValueChange: (values: z.infer<typeof FormSchema>) => void;
}
const FormContext = createContext<FormContextProps | undefined>(undefined);

export const StepperFormProvider = ({children}: {children: ReactNode}) => {
  const [stepperFormValues, setStepperFormValues] = useState<z.infer<typeof FormSchema>>({});
  const handleFormValueChange = (values: z.infer<typeof FormSchema>) => {
    setStepperFormValues(values);
  };

  return (
    <FormContext.Provider value={{stepperFormValues, handleFormValueChange}}>
      {children}
    </FormContext.Provider>
  );
};


export const useStepperForm = () => {
  const context = useContext(FormContext);
  if (context === undefined) {
    throw new Error('useFormContext must be used within a FormProvider');
  }
  return context;
};
