'use client';
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  ArrowUpIcon,
  ResetIcon,
} from '@radix-ui/react-icons';
import React, { useEffect } from 'react';

import { Button } from '@/components/ui/button';
import ExportStepMenu from '@/components/ui/export-step-menu';
import { useStepper } from '@/components/ui/stepper';
import { DownloadFileParams } from '@/lib/types';

export interface FooterProps {
  onPrevStep: () => void;
  onNextStep: () => void;
  undoLastStep: () => void;
  currentStepIndex: number;
  trueStepIndex: number;
  hasCompletedAllSteps: boolean;
  totalNSteps: number;
  userParams: DownloadFileParams;
  stepName: string;
  isLoading: boolean;
  onSubmit: (options: { useStructuredInput: boolean }) => Promise<void>;
}

export const Footer = ({
  onPrevStep: onPrevStep,
  onNextStep,
  undoLastStep,
  currentStepIndex,
  trueStepIndex,
  totalNSteps,
  userParams,
  stepName,
  isLoading,
  onSubmit,
}: FooterProps) => {
  const { setStep } = useStepper();
  useEffect(() => {
    setStep(currentStepIndex);
  }, [currentStepIndex, setStep]);

  return (
    <>
      <div className="w-full flex justify-between gap-2">
        <div className="flex gap-2">
          <Button
            disabled={currentStepIndex === 0 || isLoading}
            onClick={onPrevStep}
            size="sm"
            variant="secondary"
          >
            <ArrowLeftIcon className="mr-2" />
            Prev
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={undoLastStep}
            disabled={isLoading}
          >
            <ResetIcon className="mr-2" />
            Undo step
          </Button>
          <Button
            disabled={isLoading}
            onClick={onSubmit}
            size="sm"
            variant="default"
          >
            <ArrowUpIcon className="mr-2" />
            Run step
          </Button>
        </div>

        <ExportStepMenu
          disabled={isLoading}
          userParams={userParams}
          stepName={stepName}
        />
        <Button
          size="sm"
          onClick={onNextStep}
          disabled={
            (trueStepIndex >= totalNSteps - 1 &&
              currentStepIndex == trueStepIndex) ||
            currentStepIndex > trueStepIndex ||
            isLoading
          }
        >
          Next
          <ArrowRightIcon className="ml-2" />
        </Button>
      </div>
    </>
  );
};
