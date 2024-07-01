'use client';
import { useRouter, useSearchParams } from 'next/navigation';
import React, { useEffect } from 'react';
import { Suspense } from 'react';

import AppShell from '@/components/layout/app-shell';
import MainStepper from '@/components/main-stepper/main-stepper';
import { StepperFormProvider } from '@/contexts/stepper-form-context';
import { useUser } from '@/contexts/user-context';

function BuilderInner() {
  const { isLoggedIn, isLoading } = useUser();
  const router = useRouter();
  const searchParams = useSearchParams();
  const projectId = searchParams.get('projectId') || '';

  useEffect(() => {
    if (!isLoggedIn && !isLoading) {
      router.push('/');
    }
  }, [isLoggedIn, isLoading, router]);

  if (!projectId) {
    console.error('Missing required query parameter: projectid');
    router.push('/projects');
    return null;
  }
  return (
    <StepperFormProvider>
      <MainStepper projectId={projectId} />
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
