// Ignore unused parameter warnings coming from cppuint headers
#include <libmesh/ignore_warnings.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCase.h>
#include <libmesh/restore_warnings.h>

#include <libmesh/parallel.h>
#include <libmesh/parallel_algebra.h>

#include "test_comm.h"

using namespace libMesh;

class ParallelTest : public CppUnit::TestCase {
public:
  CPPUNIT_TEST_SUITE( ParallelTest );

  CPPUNIT_TEST( testGather );
  CPPUNIT_TEST( testAllGather );
  CPPUNIT_TEST( testBroadcast );
  CPPUNIT_TEST( testBroadcastVectorValueInt );
  CPPUNIT_TEST( testBroadcastVectorValueReal );
  CPPUNIT_TEST( testBroadcastPoint );
  CPPUNIT_TEST( testBarrier );
  CPPUNIT_TEST( testMin );
  CPPUNIT_TEST( testMax );
  CPPUNIT_TEST( testIsendRecv );
  CPPUNIT_TEST( testIrecvSend );

  CPPUNIT_TEST_SUITE_END();

private:

public:
  void setUp()
  {}

  void tearDown()
  {}



  void testGather()
  {
    std::vector<processor_id_type> vals;
    TestCommWorld->gather(0,cast_int<processor_id_type>(TestCommWorld->rank()),vals);

    if (TestCommWorld->rank() == 0)
      for (processor_id_type i=0; i<vals.size(); i++)
        CPPUNIT_ASSERT_EQUAL( i , vals[i] );
  }



  void testAllGather()
  {
    std::vector<processor_id_type> vals;
    TestCommWorld->allgather(cast_int<processor_id_type>(TestCommWorld->rank()),vals);

    for (processor_id_type i=0; i<vals.size(); i++)
      CPPUNIT_ASSERT_EQUAL( i , vals[i] );
  }



  void testBroadcast()
  {
    std::vector<unsigned int> src(3), dest(3);

    src[0]=0;
    src[1]=1;
    src[2]=2;

    if (TestCommWorld->rank() == 0)
      dest = src;

    TestCommWorld->broadcast(dest);

    for (unsigned int i=0; i<src.size(); i++)
      CPPUNIT_ASSERT_EQUAL( src[i] , dest[i] );
  }



  template <typename T>
  void testBroadcastVectorValue()
  {
    std::vector<VectorValue<T> > src(3), dest(3);

    {
      T val=T(0);
      for (unsigned int i=0; i<3; i++)
        for (unsigned int j=0; j<LIBMESH_DIM; j++)
          src[i](j) = val++;

      if (TestCommWorld->rank() == 0)
        dest = src;
    }

    TestCommWorld->broadcast(dest);

    for (unsigned int i=0; i<3; i++)
      for (unsigned int j=0; j<LIBMESH_DIM; j++)
        CPPUNIT_ASSERT_EQUAL (src[i](j), dest[i](j) );
  }



  void testBroadcastVectorValueInt()
  {
    this->testBroadcastVectorValue<int>();
  }



  void testBroadcastVectorValueReal()
  {
    this->testBroadcastVectorValue<Real>();
  }



  void testBroadcastPoint()
  {
    std::vector<Point> src(3), dest(3);

    {
      Real val=0.;
      for (unsigned int i=0; i<3; i++)
        for (unsigned int j=0; j<LIBMESH_DIM; j++)
          src[i](j) = val++;

      if (TestCommWorld->rank() == 0)
        dest = src;
    }

    TestCommWorld->broadcast(dest);

    for (unsigned int i=0; i<3; i++)
      for (unsigned int j=0; j<LIBMESH_DIM; j++)
        CPPUNIT_ASSERT_EQUAL (src[i](j), dest[i](j) );
  }



  void testBarrier()
  {
    TestCommWorld->barrier();
  }



  void testMin ()
  {
    unsigned int min = TestCommWorld->rank();

    TestCommWorld->min(min);

    CPPUNIT_ASSERT_EQUAL (min, static_cast<unsigned int>(0));
  }



  void testMax ()
  {
    processor_id_type max = TestCommWorld->rank();

    TestCommWorld->max(max);

    CPPUNIT_ASSERT_EQUAL (cast_int<processor_id_type>(max+1),
                          cast_int<processor_id_type>(TestCommWorld->size()));
  }



  void testIsendRecv ()
  {
    unsigned int procup = (TestCommWorld->rank() + 1) %
      TestCommWorld->size();
    unsigned int procdown = (TestCommWorld->size() +
                             TestCommWorld->rank() - 1) %
      TestCommWorld->size();

    std::vector<unsigned int> src_val(3), recv_val(3);

    src_val[0] = 0;
    src_val[1] = 1;
    src_val[2] = 2;

    Parallel::Request request;

    if (TestCommWorld->size() > 1)
      {
        // Default communication
        TestCommWorld->send_mode(Parallel::Communicator::DEFAULT);

        TestCommWorld->send (procup,
                             src_val,
                             request);

        TestCommWorld->receive (procdown,
                                recv_val);

        Parallel::wait (request);

        CPPUNIT_ASSERT_EQUAL ( src_val.size() , recv_val.size() );

        for (unsigned int i=0; i<src_val.size(); i++)
          CPPUNIT_ASSERT_EQUAL( src_val[i] , recv_val[i] );


        // Synchronous communication
        TestCommWorld->send_mode(Parallel::Communicator::SYNCHRONOUS);
        std::fill (recv_val.begin(), recv_val.end(), 0);

        TestCommWorld->send (procup,
                             src_val,
                             request);

        TestCommWorld->receive (procdown,
                                recv_val);

        Parallel::wait (request);

        CPPUNIT_ASSERT_EQUAL ( src_val.size() , recv_val.size() );

        for (unsigned int i=0; i<src_val.size(); i++)
          CPPUNIT_ASSERT_EQUAL( src_val[i] , recv_val[i] );

        // Restore default communication
        TestCommWorld->send_mode(Parallel::Communicator::DEFAULT);
      }
  }



  void testIrecvSend ()
  {
    unsigned int procup = (TestCommWorld->rank() + 1) %
      TestCommWorld->size();
    unsigned int procdown = (TestCommWorld->size() +
                             TestCommWorld->rank() - 1) %
      TestCommWorld->size();

    std::vector<unsigned int> src_val(3), recv_val(3);

    src_val[0] = 0;
    src_val[1] = 1;
    src_val[2] = 2;

    Parallel::Request request;

    if (TestCommWorld->size() > 1)
      {
        // Default communication
        TestCommWorld->send_mode(Parallel::Communicator::DEFAULT);

        TestCommWorld->receive (procdown,
                                recv_val,
                                request);

        TestCommWorld->send (procup,
                             src_val);

        Parallel::wait (request);

        CPPUNIT_ASSERT_EQUAL ( src_val.size() , recv_val.size() );

        for (unsigned int i=0; i<src_val.size(); i++)
          CPPUNIT_ASSERT_EQUAL( src_val[i] , recv_val[i] );

        // Synchronous communication
        TestCommWorld->send_mode(Parallel::Communicator::SYNCHRONOUS);
        std::fill (recv_val.begin(), recv_val.end(), 0);


        TestCommWorld->receive (procdown,
                                recv_val,
                                request);

        TestCommWorld->send (procup,
                             src_val);

        Parallel::wait (request);

        CPPUNIT_ASSERT_EQUAL ( src_val.size() , recv_val.size() );

        for (unsigned int i=0; i<src_val.size(); i++)
          CPPUNIT_ASSERT_EQUAL( src_val[i] , recv_val[i] );

        // Restore default communication
        TestCommWorld->send_mode(Parallel::Communicator::DEFAULT);
      }
  }

};

CPPUNIT_TEST_SUITE_REGISTRATION( ParallelTest );
