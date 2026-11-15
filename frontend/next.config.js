/** @type {import('next').NextConfig} */
// The browser only ever talks to this Next.js server; /api/* is
// proxied to the FastAPI backend, so no CORS setup is needed and
// the backend URL stays a server-side concern.
const API_URL = process.env.NEUROUTE_API_URL || 'http://localhost:8000';

module.exports = {
  async rewrites() {
    return [{ source: '/api/:path*', destination: `${API_URL}/:path*` }];
  },
};
