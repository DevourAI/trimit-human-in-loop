'use client';
import { useRouter } from 'next/navigation';
import React, { useEffect, useState } from 'react';

import AppShell from '@/components/layout/app-shell';
import {
  Popover,
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
import { WorkflowCreationForm } from '@/components/ui/workflow-creation-form';
import { useUser } from '@/contexts/user-context';
import { useUserVideosData } from '@/contexts/user-videos-context';
import { FrontendWorkflowProjection } from '@/gen/openapi/api';
import { listWorkflows } from '@/lib/api';

export default function Projects() {
  const { userData, isLoggedIn, isLoading } = useUser();
  const { userVideosData } = useUserVideosData();
  const videos = userVideosData.videos;
  const [projects, setProjects] = useState<FrontendWorkflowProjection[]>([]);
  const router = useRouter();

  useEffect(() => {
    if (!isLoggedIn && !isLoading) {
      router.push('/');
    }
  }, [isLoggedIn, isLoading, router]);

  useEffect(() => {
    async function fetchAndSetProjects() {
      const projects = await listWorkflows({ user_email: userData.email });
      // TODO project.status
      setProjects(projects);
    }
    fetchAndSetProjects();
  }, [userData]);

  async function createNewProject(
    data: z.infer<typeof WorkflowCreationFormSchema>
  ) {}

  return (
    <AppShell title="Projects">
      <div className="flex justify-center items-center h-full border-border border rounded-md">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Video</TableHead>
              <TableHead>Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {projects.map((project) => {
              return (
                <TableRow
                  key={`${project.video_hash}.${project.timeline_name}`}
                >
                  <TableCell>{project.timeline_name}</TableCell>
                  <TableCell>{project.video_hash}</TableCell>
                  <TableCell>In Progress</TableCell>
                </TableRow>
              );
            })}
            <TableRow>
              <TableCell colSpan={3}>
                <Popover>
                  <PopoverTrigger>+ New project</PopoverTrigger>
                  <PopoverContent>
                    <WorkflowCreationForm
                      isLoading={false}
                      userEmail={userData.email}
                      availableVideos={videos}
                      onSubmit={createNewProject}
                    />
                  </PopoverContent>
                </Popover>
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </div>
    </AppShell>
  );
}
