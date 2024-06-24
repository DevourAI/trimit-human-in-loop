'use client';

import Link from 'next/link';
import * as React from 'react';

import {
  FormControl,
  FormDescription,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

interface VideoFormSelectorProps {
  availableVideos: UploadedVideo[];
  onChange: (videoHash: string) => void;
  formLabel: string;
  defaultValue: string;
}
export function VideoFormSelector({
  formLabel,
  onChange,
  defaultValue,
  availableVideos,
}: VideoFormSelectorProps) {
  return (
    <FormItem>
      <FormLabel>Video</FormLabel>
      <Select onValueChange={onChange} defaultValue={defaultValue}>
        <FormControl>
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
        </FormControl>
        <SelectContent>
          {availableVideos !== null && availableVideos !== undefined ? (
            availableVideos.map((video) => {
              return (
                <SelectItem key={video.video_hash} value={video.video_hash}>
                  {video.filename}
                </SelectItem>
              );
            })
          ) : (
            <SelectItem key="tmp"></SelectItem>
          )}
        </SelectContent>
      </Select>
      <FormDescription>
        You can upload new videos in the
        <Link href="/videos">videos tab</Link>.
      </FormDescription>
      <FormMessage />
    </FormItem>
  );
}
