"use client";
import useSWR from 'swr'
import React, { createContext, useContext, useState } from 'react';
import MainNav from "@/components/main-nav"
import {
  PageActions,
  PageHeader,
  PageHeaderDescription,
  PageHeaderHeading,
} from "@/components/page-header"
import { cn } from "@/lib/utils"
import MainStepper from "@/components/main-stepper"
import Login from '@/components/login';

const fetcher = (url) => fetch(url).then((res) => res.json());

export default function Home() {
  const { data, error, isLoading } = useSWR("/userData", fetcher);
  let initialUserData = {'email': '', 'name': '', 'picture': ''};
  if (data && data.userData && data.userData.value) {
    initialUserData = data.userData.value;
  }
  const [userData, setUserData] = useState(initialUserData)

  return (
      <div className="container relative">
        <MainNav />
        <PageHeader>
          <PageHeaderHeading>TrimIt Interview Builder</PageHeaderHeading>
          <PageHeaderDescription>
            Raw interview footage to edited video, no timeline required.
          </PageHeaderDescription>
          <PageActions>
            <Login userData={userData} setUserData={setUserData}/>
          </PageActions>
        </PageHeader>
        <MainStepper userData={userData}/>
      </div>
  )
}
