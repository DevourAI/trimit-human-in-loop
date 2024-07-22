'use client';
import { useRouter } from 'next/navigation';
import React, { useEffect, useState } from 'react';
import { z } from 'zod';

import AppShell from '@/components/layout/app-shell';
import { Button } from '@/components/ui/button';
import {
  Popover,
  PopoverAnchor,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  WorkflowCreationForm,
  WorkflowCreationFormSchema,
} from '@/components/ui/workflow-creation-form';
import { useUser } from '@/contexts/user-context';
import { useUserVideosData } from '@/contexts/user-videos-context';
import { FrontendWorkflowProjection } from '@/gen/openapi/api';
import { createNewWorkflow, listWorkflows } from '@/lib/api';
import { CreateNewWorkflowParams } from '@/lib/types';

export default function Projects() {
  const { userData, isLoggedIn, isLoading } = useUser();
  const { userVideosData } = useUserVideosData();
  const videos = userVideosData.videos;
  const [workflows, setWorkflows] = useState<FrontendWorkflowProjection[]>([]);
  const [latestWorkflowId, setLatestWorkflowId] = useState<string>('');
  const [selectedWorkflow, setSelectedWorkflow] =
    useState<FrontendWorkflowProjection | null>(null);
  const router = useRouter();

  useEffect(() => {
    if (selectedWorkflow?.id) {
      router.push(`/builder?workflowId=${selectedWorkflow.id}`);
    }
  }, [router, selectedWorkflow]);

  useEffect(() => {
    if (!isLoggedIn && !isLoading) {
      router.push('/');
    }
  }, [isLoggedIn, isLoading, router]);

  useEffect(() => {
    async function fetchAndSetProjects() {
      const workflows = await listWorkflows({ user_email: userData.email });
      // TODO project.status
      setWorkflows(workflows);
    }
    if (userData.email) {
      fetchAndSetProjects();
    }
  }, [userData, latestWorkflowId]);

  async function createNewProjectWrapper(
    data: z.infer<typeof WorkflowCreationFormSchema>
  ) {
    const workflowId = await createNewWorkflow({
      user_email: userData.email,
      ...data,
    } as CreateNewWorkflowParams);
    setLatestWorkflowId(workflowId);
  }

  return (
    <AppShell title="Projects">
      <Popover>
        <div className="flex justify-center items-center h-full border-border border rounded-md">
          <PopoverAnchor />
          <PopoverContent>
            <WorkflowCreationForm
              isLoading={false}
              userEmail={userData.email}
              availableVideos={videos}
              onSubmit={createNewProjectWrapper}
            />
          </PopoverContent>
        </div>

        <div className="flex justify-center items-center h-full border-border border rounded-md">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Project Name</TableHead>
                <TableHead>Timeline Name</TableHead>
                <TableHead>Video</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Select</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {workflows.map((workflow) => {
                return (
                  <TableRow
                    key={`${workflow.video_hash}.${workflow.timeline_name}`}
                  >
                    <TableCell>{workflow.project_name}</TableCell>
                    <TableCell>{workflow.timeline_name}</TableCell>
                    <TableCell>{workflow.video_hash}</TableCell>
                    <TableCell>In Progress</TableCell>
                    <TableCell>
                      <Button
                        variant="outline"
                        onClick={() => setSelectedWorkflow(workflow)}
                      >
                        Select
                      </Button>
                    </TableCell>
                  </TableRow>
                );
              })}
              <TableRow>
                <TableCell colSpan={3}>
                  <PopoverTrigger>+ New project</PopoverTrigger>
                </TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </div>
      </Popover>
    </AppShell>
  );
}
