'use client';
import { useRouter } from 'next/navigation';
import React, { useEffect, useState } from 'react';

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
import { WorkflowCreationForm } from '@/components/ui/workflow-creation-form';
import { useUser } from '@/contexts/user-context';
import { useUserVideosData } from '@/contexts/user-videos-context';
import { FrontendWorkflowProjection } from '@/gen/openapi/api';
import { createNewWorkflow, listWorkflows } from '@/lib/api';
import { CreateNewWorkflowParams } from '@/lib/types';

export default function Projects() {
  const { userData, isLoggedIn, isLoading } = useUser();
  const { userVideosData } = useUserVideosData();
  const videos = userVideosData.videos;
  const [projects, setProjects] = useState<FrontendWorkflowProjection[]>([]);
  const [selectedProject, setSelectedProject] =
    useState<FrontendWorkflowProjection | null>(null);
  const router = useRouter();

  useEffect(() => {
    if (selectedProject?.id) {
      console.log(selectedProject);
      router.push(`/builder?projectId=${selectedProject.id}`);
    }
  }, [router, selectedProject]);

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
    if (userData.email) {
      fetchAndSetProjects();
    }
  }, [userData]);

  async function createNewProjectWrapper(
    data: z.infer<typeof WorkflowCreationFormSchema>
  ) {
    await createNewWorkflow({
      email: userData.email,
      ...data,
    } as CreateNewWorkflowParams);
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
                <TableHead>Name</TableHead>
                <TableHead>Video</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Select</TableHead>
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
                    <TableCell>
                      <Button
                        variant="outline"
                        onClick={() => setSelectedProject(project)}
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
