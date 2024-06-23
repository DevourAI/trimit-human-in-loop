'use client';
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  ResetIcon,
} from '@radix-ui/react-icons';
import React, { useEffect } from 'react';

import { Button } from '@/components/ui/button';
import { useStepper } from '@/components/ui/stepper';

export interface FooterProps {
  onPrevStep: () => void;
  onNextStep: () => void;
  undoLastStep: () => void;
  currentStepIndex: number;
  trueStepIndex: number;
  hasCompletedAllSteps: boolean;
  totalNSteps: number;
}

export const Footer = ({
  onPrevStep: onPrevStep,
  onNextStep,
  undoLastStep,
  currentStepIndex,
  trueStepIndex,
  hasCompletedAllSteps,
  totalNSteps,
}: FooterProps) => {
  const {
    nextStep,
    prevStep,
    resetSteps,
    isDisabledStep,
    isLastStep,
    isOptionalStep,
    setStep,
  } = useStepper();
  useEffect(() => {
    setStep(currentStepIndex);
  }, [currentStepIndex, setStep]);

  return (
    <>
      {hasCompletedAllSteps && (
        <div className="h-40 flex items-center justify-center my-4 border bg-secondary text-primary rounded-md">
          <h3 className="text-xl">
            TrimIt finished editing your video, but feel free to provide
            additional feedback or go back to previous steps
          </h3>
        </div>
      )}
      <div className="w-full flex justify-between gap-2">
        <div className="flex gap-2">
          <Button
            disabled={currentStepIndex === 0}
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
            disabled={currentStepIndex === 0}
          >
            <ResetIcon className="mr-2" />
            Undo step
          </Button>
        </div>
        <Button
          size="sm"
          onClick={onNextStep}
          disabled={
            trueStepIndex >= totalNSteps - 1 ||
            currentStepIndex >= trueStepIndex
          }
        >
          Next
          <ArrowRightIcon className="ml-2" />
        </Button>
      </div>
    </>
  );
};
