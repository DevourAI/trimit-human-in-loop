"use client";
import React, { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useUser } from "@/contexts/user-context";
import VideoSelector from "@/components/ui/video-selector";
import AppShell from "@/components/layout/app-shell";

export default function Videos() {
  const { isLoggedIn, isLoading } = useUser();
  const router = useRouter();

  useEffect(() => {
    if (!isLoggedIn && !isLoading) {
      router.push("/");
    }
  }, [isLoggedIn, isLoading, router]);

  const handleVideoSelected = (videoHash: string) => {
    // TODO: We should "create a new project" with the selected video here and redirect to it instead
    router.push(`/builder?videoHash=${videoHash}`);
  };

  return (
    <AppShell title="Videos" subtitle="Select a video to get started">
      <VideoSelector setVideoHash={handleVideoSelected} />
    </AppShell>
  );
}
