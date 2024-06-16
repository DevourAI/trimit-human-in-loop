'use client';
import React, { useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import MainStepper from '@/components/main-stepper/main-stepper';
import AppShell from '@/components/layout/app-shell';
import { useUser } from '@/contexts/user-context';

export default function Builder() {
  const { isLoggedIn, isLoading } = useUser();
  const router = useRouter();
  const searchParams = useSearchParams();
  const videoHash = searchParams.get('videoHash');

  useEffect(() => {
    if (!isLoggedIn && !isLoading) {
      router.push('/');
    }
  }, [isLoggedIn, isLoading, router]);

  if (!videoHash) {
    console.error('Missing required query parameter: videoHash');
    router.push('/videos');
    return null;
  }

  return (
    <AppShell title="Builder">
      <MainStepper videoHash={videoHash} />
    </AppShell>
  );
}
