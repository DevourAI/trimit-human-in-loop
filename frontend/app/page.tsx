'use client';
import { useRouter } from 'next/navigation';
import React, { useEffect } from 'react';

import AppShell from '@/components/layout/app-shell';
import {
  PageActions,
  PageHeader,
  PageHeaderDescription,
  PageHeaderHeading,
} from '@/components/layout/page-header';
import Login from '@/components/login';
import { useUser } from '@/contexts/user-context';

export default function Home() {
  const { userData, setUserData, isLoggedIn } = useUser();
  const router = useRouter();

  useEffect(() => {
    if (isLoggedIn) {
      router.push('/videos');
    }
  }, [isLoggedIn, router]);

  return (
    <AppShell>
      <PageHeader>
        <PageHeaderHeading>TrimIt Interview Builder</PageHeaderHeading>
        <PageHeaderDescription>
          Raw interview footage to edited video, no timeline required.
        </PageHeaderDescription>
        <PageActions>
          {!isLoggedIn && (
            <Login userData={userData} setUserData={setUserData} />
          )}
        </PageActions>
      </PageHeader>
    </AppShell>
  );
}
