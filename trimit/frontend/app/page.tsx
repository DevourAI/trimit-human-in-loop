"use client";
import Image from "next/image";
import Link from "next/link";
import { GetServerSideProps } from 'next';
import useSWR from 'swr'
import { Button, buttonVariants } from "@/components/ui/button"
import MainNav from "@/components/main-nav"
import {
  PageActions,
  PageHeader,
  PageHeaderDescription,
  PageHeaderHeading,
} from "@/components/page-header"
import { cn } from "@/lib/utils"
import MainStepper from "@/components/main-stepper"
import { useSearchParams } from 'next/navigation';

import { googleLogout, GoogleLogin, GoogleOAuthProvider } from '@react-oauth/google';
import React, { createContext, useContext, useState } from 'react';



const fetcher = (url) => fetch(url).then((res) => res.json());
const fetchWithToken = (url, token) => {
  const headers = {
    Authorization: `Bearer ${token}`,
  };
  return fetch(url, {headers: headers}).then((res) => res.json());
}
const fetchWithBody = (url, param) => {
  return fetch(url, {body: param}).then((res) => res.json());
}
const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
const clientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;

const getProtectedEndpoint = (url: string, token: string) => {
  const { data, error, isLoading } = useSWR(
    [`${baseUrl}${url}`, token],
    ([url, token]) => fetchWithToken(url, token),
  )
  if (!error && !isLoading) {
    console.log(data);
    return data;
  } else {
    console.log(error, isLoading);
  }
}



export default function Home() {
  const [token, setToken] = useState("")
  const logout = () => {
    console.log("logging out");
    googleLogout();
    setToken("");
  }
  if (!token) {
    const { data, error, isLoading } = useSWR<string>("/cookies", fetcher);
    console.log("data", data);
    console.log("error", error);
    // if (data.token) {
      // setToken(data.token);
    // }
  }



  let loggedIn = token ? true : false;
  console.log('loggedIn', loggedIn);


  // TODO somehow GoogleOAuthProvider is causing an error when the thing rerenders
  return (
      <div className="container relative">
        <MainNav />
        <PageHeader>
          <PageHeaderHeading>Build your component library</PageHeaderHeading>
          <PageHeaderDescription>
            Beautifully designed components that you can copy and paste into your
            apps. Accessible. Customizable. Open Source.
          </PageHeaderDescription>
          <PageActions>
            <Button onClick={() => getProtectedEndpoint("/getUserData")} className={cn(buttonVariants())}>
              Get User Data
            </Button>

              <GoogleOAuthProvider clientId={clientId}>
                <GoogleLogin
                   onSuccess={credentialResponse => {
                     console.log(credentialResponse);
                     setToken(credentialResponse.credential);
                     // fetch(`/login?token=${credentialResponse.credential}`).then((res) => {
                       // console.log(res);
                     // });
                   }}
                   onError={() => {
                     console.log('Login Failed');
                   }}
                   useOneTap
                 />;
              </GoogleOAuthProvider>
            { loggedIn?
              <Button onClick={() => logout()} className={cn(buttonVariants())}>
                Logout
              </Button>
            : null
            }
          </PageActions>
        </PageHeader>
        <MainStepper />
      </div>
  )
}
