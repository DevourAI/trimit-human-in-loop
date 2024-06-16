'use client';
import React from 'react';
import AppShell from '@/components/layout/app-shell';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';

export default function Projects() {
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
            <TableRow>
              <TableCell>Project Alpha</TableCell>
              <TableCell>Video 1</TableCell>
              <TableCell>In Progress</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Project Beta</TableCell>
              <TableCell>Video 2</TableCell>
              <TableCell>Completed</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Project Gamma</TableCell>
              <TableCell>Video 3</TableCell>
              <TableCell>Not Started</TableCell>
            </TableRow>
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
