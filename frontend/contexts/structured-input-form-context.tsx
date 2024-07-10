import React, {ReactNode, createContext, useContext, useState, useEffect, useRef} from 'react';
import {zodResolver} from '@hookform/resolvers/zod';
import type {UseFormReturn} from "react-hook-form";
import {useForm} from "react-hook-form";
import {z} from 'zod';
import {getFunctionCallResults, redoExportResults, getLatestExportResults} from '@/lib/api';
import {FrontendStepOutput} from '@/gen/openapi';
import {DownloadFileParams} from '@/lib/types';
const POLL_INTERVAL = 5000;

// TODO autogenerate these based on types in @/gen/openapi/api.ts
export const RemoveOffScreenSpeakersFormSchema = z.object({
  speaker_name_mapping: z.record(z.string(), z.string()),
  speaker_tag_mapping: z.record(z.string(), z.boolean()),
});
export const IdentifyKeySoundbitesInput = z.object({
  soundbite_selection: z.record(z.string(), z.boolean()),
});

export const StructuredInputFormSchema = z.object({
  remove_off_screen_speakers: RemoveOffScreenSpeakersFormSchema.optional(),
  identify_key_soundbites: IdentifyKeySoundbitesInput.optional()
});

interface FormContextProps {
  form: UseFormReturn;
  exportResult: Record<string, any> | null;
}

interface StructuredInputFormProviderProps {
  children: ReactNode;
  stepOutput: FrontendStepOutput | null;
  onFormDataChange: (values: z.infer<typeof StructuredInputFormSchema>) => void;
  userParams: DownloadFileParams
}

function setUndefinedToTrue(values: Record<string | number, any>) {
  return Object.keys(values).reduce(
    (acc: any, key: string) => {
      acc[key] = (acc[key] === undefined || acc[key] === null) ? true : acc[key];
      return acc;
    },
    {}
  );
}


function createStructuredInputDefaultsFromOutputs(stepOutput: FrontendStepOutput | null, exportResult: Record<string, any> | null) {
  let defaultNameMapping = {};
  let defaultTagMapping = {};
  let defaultSoundbiteSelectionMapping = {};
  if (exportResult) {
    const speakerTaggingClips = exportResult.speaker_tagging_clips || {};
    defaultNameMapping = Object.keys(speakerTaggingClips).reduce(
      (acc: any, key) => {
        acc[key] = '';
        return acc;
      },
      {}
    );
    defaultTagMapping = Object.keys(speakerTaggingClips).reduce(
      (acc: any, key) => {
        // TODO set this based on actual output values (on_screen_speakers)
        acc[key] = true;
        return acc;
      },
      {}
    );
    const soundbiteClips = stepOutput?.step_outputs?.current_soundbites_state?.soundbites || [];
    defaultSoundbiteSelectionMapping = soundbiteClips.reduce(
      (acc: any, _: string, index: number) => {
        acc[index] = true;
        return acc;
      },
      {}
    );
  }
  return {
    remove_off_screen_speakers: {
      speaker_name_mapping: defaultNameMapping,
      speaker_tag_mapping: defaultTagMapping,
    },
    identify_key_soundbites: {
      soundbite_selection: defaultSoundbiteSelectionMapping
    }
  }
}

export const StructuredInputFormProvider: React.FC<StructuredInputFormProviderProps> = ({children, stepOutput, userParams, onFormDataChange}) => {
  const [exportCallId, setExportCallId] = useState<string>('');
  const newExportCallId = useRef<boolean>(false);
  useEffect(() => {
    if (stepOutput?.export_call_id && !newExportCallId.current) {
      console.log("setting export call id to stepOutput", stepOutput?.export_call_id);
      setExportCallId(stepOutput.export_call_id);
    }
  }, [stepOutput?.export_call_id]);
  const [exportResult, setExportResult] = useState<Record<string, any> | null>(
    stepOutput?.export_result || null
  );
  const exportResultDone = useRef<boolean>(
    stepOutput?.export_result !== undefined && stepOutput?.export_result !== null && Object.keys(stepOutput.export_result).length > 0
  );
  const timeoutId = useRef<NodeJS.Timeout | null>(null);
  const isComponentMounted = useRef<boolean>(true);
  const isPolling = useRef<boolean>(false);

  useEffect(() => {
    async function checkAndSetExportResultStatus() {
      if (isPolling.current) return; // Ensure only one polling request in flight
      isPolling.current = true;
      console.log('exportCallId', exportCallId);
      const statuses: Array<any> = exportCallId.length > 0 ? (await getFunctionCallResults([exportCallId]) as Array<any>) : [] as Array<any>;
      if (statuses[0] && statuses[0].status === 'done') {
        const newExportResults = await getLatestExportResults({step_name: stepOutput?.step_name, ...userParams});
        console.log("got new export results", newExportResults);
        setExportResult(newExportResults);
        exportResultDone.current = true;
      } else if (statuses[0] && statuses[0].status == 'error') {
        console.log('export error:', statuses[0]);
        await redoExportResults({step_name: stepOutput?.step_name, ...userParams}).then((result) => {
          console.log("got redo export result call id:", result);
          setExportCallId(result);
          newExportCallId.current = true;
        });
        setExportResult(null);
      }
      isPolling.current = false;
    }

    const pollForStatuses = async () => {
      await checkAndSetExportResultStatus();
      if (isComponentMounted.current && !exportResultDone.current) {
        timeoutId.current = setTimeout(pollForStatuses, POLL_INTERVAL);
      }
    };

    if (stepOutput?.export_call_id && !exportResultDone.current) {
      pollForStatuses();
    } else if (stepOutput?.export_result) {
      setExportResult(stepOutput.export_result);
    }
    return () => {
      if (timeoutId.current) {
        clearTimeout(timeoutId.current);
      }
    };
  }, [exportCallId, stepOutput?.export_result]);



  const defaultValues = useRef(createStructuredInputDefaultsFromOutputs(stepOutput, exportResult));

  const form = useForm<z.infer<typeof StructuredInputFormSchema>>(
    {
      resolver: zodResolver(StructuredInputFormSchema),
      defaultValues: defaultValues.current,
    }

  );

  const prevFormDataString = useRef<string>();

  useEffect(() => {
    let currentValues = form.getValues();
    const newDefaultValues = createStructuredInputDefaultsFromOutputs(stepOutput, exportResult);
    if (
      JSON.stringify(currentValues) !== prevFormDataString.current
      || JSON.stringify(defaultValues.current) !== JSON.stringify(newDefaultValues)
    ) {

      defaultValues.current = newDefaultValues;
      if (currentValues.identify_key_soundbites?.soundbite_selection) {
        const soundbite_selection = currentValues.identify_key_soundbites?.soundbite_selection;
        if (Object.keys(soundbite_selection).length > 0) {
          currentValues.identify_key_soundbites.soundbite_selection = setUndefinedToTrue(
            currentValues.identify_key_soundbites.soundbite_selection
          );
        } else {
          currentValues.identify_key_soundbites = defaultValues.current.identify_key_soundbites;
        }
      } else {
        currentValues.identify_key_soundbites = defaultValues.current.identify_key_soundbites;
      }
      onFormDataChange(currentValues);
      prevFormDataString.current = JSON.stringify(currentValues);
    }
  }, [form.watch(), onFormDataChange, exportResult]);

  return (
    <StructuredInputFormContext.Provider value={{form, exportResult}}>
      {children}
    </StructuredInputFormContext.Provider>
  );
};

const StructuredInputFormContext = createContext<FormContextProps | undefined>(undefined);

export const useStructuredInputForm = () => {
  const context = useContext(StructuredInputFormContext);
  if (context === undefined) {
    throw new Error('useStructuredInputFormContext must be used within a StructuredInputFormProvider');
  }
  return context;
};
