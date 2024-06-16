'use client';
import React from 'react';
import { useParams } from 'next/navigation';
import AppShell from '@/components/layout/app-shell';

export default function Project() {
  const { id } = useParams();

  return (
    <AppShell title={`Project ${id}`}>
      {/* TODO: Fetch and render project details by ID and render stepper */}
      <p>Project ID: {id}</p>
    </AppShell>
  );
}
