'use client';
import React, { useEffect, useState } from 'react';

import AppShell from '@/components/layout/app-shell';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useUser } from '@/contexts/user-context';
import { FrontendWorkflowProjection } from '@/gen/openapi/api';
import { listWorkflows } from '@/lib/api';

export default function Projects() {
  const { userData } = useUser();
  const [projects, setProjects] = useState<FrontendWorkflowProjection[]>([]);
  useEffect(() => {
    async function fetchAndSetProjects() {
      const projects = await listWorkflows({ user_email: userData.email });
      // TODO project.status
      setProjects(projects);
    }
    fetchAndSetProjects();
  }, [userData]);

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
                <a href="/videos">
                  <Button variant="ghost">+ New project</Button>
                </a>
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </div>
    </AppShell>
  );
}
