'use client';
import { useRouter, useSearchParams } from 'next/navigation';
import React, { useEffect, useState } from 'react';
import { Suspense } from 'react';

import AppShell from '@/components/layout/app-shell';
import MainStepper from '@/components/main-stepper/main-stepper';
import OneButtonGenerate from '@/components/one-button-generate';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { StepperFormProvider } from '@/contexts/stepper-form-context';
import { useUser } from '@/contexts/user-context';

function BuilderInner() {
  const { isLoggedIn, isLoading } = useUser();
  const router = useRouter();
  const searchParams = useSearchParams();
  const workflowId = searchParams.get('workflowId') || '';
  const projectName = searchParams.get('projectName') || '';
  const projectId = searchParams.get('projectId') || '';
  const [easyMode, setEasyMode] = useState<boolean>(true);

  useEffect(() => {
    if (!isLoggedIn && !isLoading) {
      router.push('/');
    }
  }, [isLoggedIn, isLoading, router]);

  const onCheckedChange = () => {
    if (easyMode) {
      setEasyMode(false);
      if (!workflowId) {
        router.push('/projects');
      }
    } else {
      setEasyMode(true);
    }
  };
  return (
    <StepperFormProvider>
      <Switch
        id="easy-mode"
        checked={easyMode}
        onCheckedChange={onCheckedChange}
      />
      <Label htmlFor="easy-mode">Easy Mode</Label>
      {easyMode ? (
        <OneButtonGenerate
          initialProjectId={projectId}
          initialProjectName={projectName}
          initialWorkflowId={workflowId}
        />
      ) : (
        <MainStepper
          initialProjectId={projectId}
          initialProjectName={projectName}
          workflowId={workflowId}
        />
      )}
    </StepperFormProvider>
  );
}

export default function Builder() {
  return (
    <AppShell title="Builder">
      <Suspense>
        <BuilderInner />
      </Suspense>
    </AppShell>
  );
}
