import Image from "next/image";
import Link from "next/link";
import { GetServerSideProps } from 'next';
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
import Login from '@/components/login';

export default function Home() {
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
            <Login />
          </PageActions>
        </PageHeader>
        <MainStepper />
      </div>
  )
}
