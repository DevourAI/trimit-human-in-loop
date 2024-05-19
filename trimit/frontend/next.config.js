/** @type {import('next').NextConfig} */
require('dotenv').config({ path: `./.env.${process.env.NODE_ENV}`});
const nextConfig = {
  rewrites: async () => {
    return [
      {
        source: "/api/:path*",
        destination:
          process.env.NODE_ENV === "development"
            ? "http://127.0.0.1:8000/api/:path*"
            : `${process.env.NODE_ENV}/api/`,
      },
    ];
  },
};

module.exports = nextConfig;
