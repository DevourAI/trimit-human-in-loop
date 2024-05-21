"use client";
import { jwtDecode } from "jwt-decode";
import { cn } from "@/lib/utils"
import { Button, buttonVariants } from "@/components/ui/button"
import { googleLogout, GoogleLogin } from '@react-oauth/google';
import React, { createContext, useContext, useState } from 'react';
import { GoogleOAuthProvider } from '@react-oauth/google';

const clientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;

export default function Login({userData, setUserData}) {
  const logout = () => {
    googleLogout();
    setUserData({'email': '', 'name': '', 'picture': ''})
  }
  let loggedIn = userData.email ? true : false;

  return (
    <div>
        <GoogleOAuthProvider clientId={clientId}>

            { loggedIn?
            <h1>Welcome {userData.given_name}</h1>
            : null
            }

            { loggedIn?
              <Button onClick={() => logout()} className={cn(buttonVariants())}>
                Logout
              </Button>
            :
                <GoogleLogin
                   onSuccess={credentialResponse => {
                     const decoded = jwtDecode(credentialResponse.credential);
                     setUserData(decoded);
                     fetch(`/userData`, {
                       body: JSON.stringify({userData: decoded}),
                       method: 'POST',
                       headers: {
                         'Content-Type': 'application/json'
                       }
                     }).then((res) => {});
                   }}
                   onError={() => {
                     console.log('Login Failed');
                   }}
                   useOneTap
                   auto_select
                 />
            }
        </GoogleOAuthProvider>
     </div>
   )
}
