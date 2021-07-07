module fruit_mpi
  use fruit
  use mpi
  implicit none
  private

  integer, parameter :: XML_OPEN = 20
  integer, parameter :: XML_WORK = 21
  character(len = *), parameter :: xml_filename = "result.xml"
  integer, parameter :: NUMBER_LENGTH = 10
  integer, parameter :: FN_LENGTH = 50

  public ::          fruit_init_mpi_xml
  interface          fruit_init_mpi_xml
    module procedure fruit_init_mpi_xml_
  end interface

  public ::          fruit_finalize_mpi
  interface          fruit_finalize_mpi
    module procedure fruit_finalize_mpi_
  end interface

  public ::          fruit_summary_mpi
  interface          fruit_summary_mpi
    module procedure fruit_summary_mpi_
  end interface

  public ::          fruit_summary_mpi_xml
  interface          fruit_summary_mpi_xml
    module procedure fruit_summary_mpi_xml_
  end interface
contains
  subroutine fruit_init_mpi_xml_(rank)
    integer, intent(in) :: rank
    character(len = FN_LENGTH) :: xml_filename_work

    write(xml_filename_work, '("result_tmp_", i5.5, ".xml")') rank
    call set_xml_filename_work(xml_filename_work)

    call init_fruit_xml(rank)
  end subroutine fruit_init_mpi_xml_

  subroutine     fruit_finalize_mpi_(size, rank)
    integer, intent(in) :: size, rank
    if (size < 0) print *, "size negative"
    if (rank < 0) print *, "rank negative"
    call         fruit_finalize
  end subroutine fruit_finalize_mpi_

  subroutine     fruit_summary_mpi_(size, rank)
    integer, intent(in) :: size, rank
    integer :: fail_assert_sum
    integer :: succ_assert_sum
    integer :: fail_case_sum
    integer :: succ_case_sum

    integer :: fail_assert
    integer :: succ_assert
    integer :: fail_case
    integer :: succ_case

    integer :: message_index
    integer :: num_msgs
    integer :: num_msgs_sum
    integer, allocatable :: num_msgs_rank(:)

    integer :: ierr
    integer :: i
    integer :: imsg
    integer :: status(MPI_STATUS_SIZE)

    integer, parameter :: MSG_LENGTH_HERE = 256
    character(len = MSG_LENGTH_HERE), allocatable :: msgs(:)
    character(len = MSG_LENGTH_HERE), allocatable :: msgs_all(:)

    call get_assert_and_case_count(&
    & fail_assert,  succ_assert, &
    & fail_case,    succ_case)

    call get_message_index(message_index)
    num_msgs = message_index - 1
    allocate(msgs(num_msgs))
    call get_message_array(msgs)

    allocate(num_msgs_rank(size))
    call MPI_Allgather(&
    & num_msgs,      1, MPI_INTEGER, &
    & num_msgs_rank, 1, MPI_INTEGER, MPI_COMM_WORLD, ierr)

    num_msgs_sum = sum(num_msgs_rank(:))
    allocate(msgs_all(num_msgs_sum))

    ! array msgs_all:
    !
    ! | msgs(:) of rank 0  | msgs(:) of rank 1   | msgs(:) of rank 2  |
    ! |                    |                     |                    |
    ! | num_msgs_rank(1)   |  num_msgs_rank(2)   | num_msgs_rank(3)   |
    ! |                    |                     |                    |
    ! |                    |                     |                    |
    !                       A                     A                  A
    !                       |                     |                  |
    !              sum(num_msgs_rank(1:1))+1      |             num_msgs_sum
    !                                    sum(num_msgs_rank(1:2))+1

    if (rank == 0) then
      msgs_all(1:num_msgs) = msgs(1:num_msgs)
      do i = 1, size - 1
        imsg = sum(num_msgs_rank(1:i)) + 1
        call MPI_RECV(&
        & msgs_all(imsg), &
        & num_msgs_rank(i + 1) * MSG_LENGTH_HERE, MPI_CHARACTER, &
        & i, 7, MPI_COMM_WORLD, status, ierr)
      enddo
    else
      call MPI_Send(&
      & msgs, &
      & num_msgs * MSG_LENGTH_HERE               , MPI_CHARACTER, &
      & 0, 7, MPI_COMM_WORLD, ierr)
    endif

    call MPI_REDUCE(&
    & fail_assert    , &
    & fail_assert_sum, 1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

    call MPI_REDUCE(&
    & succ_assert    , &
    & succ_assert_sum, 1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

    call MPI_REDUCE(&
    & fail_case    , &
    & fail_case_sum,   1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

    call MPI_REDUCE(&
    & succ_case    , &
    & succ_case_sum,   1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

    if (rank == 0) then
      write (*,*)
      write (*,*)
      write (*,*) '    Start of FRUIT summary: '
      write (*,*)

      if (fail_assert_sum > 0) then
         write (*,*) 'Some tests failed!'
      else
         write (*,*) 'SUCCESSFUL!'
      end if

      write (*,*)
      write (*,*) '  -- Failed assertion messages:'

      do i = 1, num_msgs_sum
        write (*, "(A)") '   '//trim(msgs_all(i))
      end do

      write (*, *) '  -- end of failed assertion messages.'
      write (*, *)

      if (succ_assert_sum + fail_assert_sum /= 0) then
        call fruit_summary_table(&
        & succ_assert_sum, fail_assert_sum, &
        & succ_case_sum  , fail_case_sum    &
        &)
      endif
      write(*, *) '  -- end of FRUIT summary'
    endif
  end subroutine fruit_summary_mpi_

  subroutine     fruit_summary_mpi_xml_(size, rank)
    integer, intent(in) :: size, rank

    character(len = 1000) :: whole_line
    character(len =  100) :: full_count
    character(len =  100) :: fail_count
    character(len = FN_LENGTH)              :: xml_filename_work
    character(len = FN_LENGTH), allocatable :: xml_filename_work_all(:)

    integer :: fail_assert    , succ_assert    , fail_case     , succ_case
    integer :: fail_assert_sum, succ_assert_sum, fail_case_sum , succ_case_sum
    integer :: i
    integer :: status(MPI_STATUS_SIZE)
    integer :: ierr

    call get_xml_filename_work(xml_filename_work)

    allocate(xml_filename_work_all(size))

    if (rank /= 0) then
    call MPI_Send(    xml_filename_work, &
    &     FN_LENGTH, MPI_CHARACTER,     0, 8, MPI_COMM_WORLD, ierr)
    endif
    if (rank == 0) then
      xml_filename_work_all(1) = xml_filename_work

      do i = 1+1, size
        call MPI_RECV(xml_filename_work_all(i), &
        & FN_LENGTH, MPI_CHARACTER, i - 1, 8, MPI_COMM_WORLD, status, ierr)
      enddo
    endif

    call get_assert_and_case_count(&
    & fail_assert,  succ_assert, &
    & fail_case,    succ_case)

    call MPI_REDUCE(&
    & fail_assert    , &
    & fail_assert_sum, 1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

    call MPI_REDUCE(&
    & succ_assert    , &
    & succ_assert_sum, 1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

    call MPI_REDUCE(&
    & fail_case    , &
    & fail_case_sum,   1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

    call MPI_REDUCE(&
    & succ_case    , &
    & succ_case_sum,   1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

    full_count = int_to_str(succ_case_sum + fail_case_sum)
    fail_count = int_to_str(                fail_case_sum)

    if (rank == 0) then
      open (XML_OPEN, file = xml_filename, action ="write", status = "replace")
        write(XML_OPEN, '("<?xml version=""1.0"" encoding=""UTF-8""?>")')
        write(XML_OPEN, '("<testsuites>")')
        write(XML_OPEN, '("  <testsuite errors=""0"" ")', advance = "no")
        write(XML_OPEN, '("tests=""", a, """ ")', advance = "no") &
        &  trim(full_count)
        write(XML_OPEN, '("failures=""", a, """ ")', advance = "no") &
        &  trim(fail_count)
        write(XML_OPEN, '("name=""", a, """ ")', advance = "no") &
        &  "name of test suite"
        write(XML_OPEN, '("id=""1"">")')

        do i = 1, size
          open (XML_WORK, FILE = xml_filename_work_all(i))
            do
              read(XML_WORK, '(a)', end = 999) whole_line
              write(XML_OPEN, '(a)') trim(whole_line)
            enddo
        999 continue
          close(XML_WORK)
        enddo

        write(XML_OPEN, '("  </testsuite>")')
        write(XML_OPEN, '("</testsuites>")')
      close(XML_OPEN)
    endif
    if (size < 0) print *, "size < 0"
  end subroutine fruit_summary_mpi_xml_

  function int_to_str(i)
    integer, intent(in) :: i
    character(LEN = NUMBER_LENGTH) :: int_to_str

    write(int_to_str, '(i10)') i
    int_to_str = adjustl(int_to_str)
  end function int_to_str
end module fruit_mpi
