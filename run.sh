#!/bin/sh

# Constants
BOOL_FALSE=0
BOOL_TRUE=1
HUGE_PAGES_DEBIAN_PATH="/sys/kernel/mm/transparent_hugepage/enabled"
HUGE_PAGES_REDHAT_PATH="/sys/kernel/mm/redhat_transparent_hugepage/enabled"

# Set default value
LD_PRELOAD=
VERBOSE=$BOOL_FALSE
ENABLE_HUGEPAGE=$BOOL_FALSE

# Add module sources to PYTHON
PYTHONPATH=src

function print(){
    if [ "$VERBOSE" -eq "$BOOL_TRUE" ]; then
      echo -e "$1"
    fi
}

function set_hugepage_status() {
#  Enable or  disable the use of Huge Page when allocating large block of memory.
# Huge Page usually maps memory blocks of 2Mo rather 4Kb which results in less blocks to be
# translated by the MMU (Memory Mapping Unit) thus increasing the probability of CPU cache hits.
#
# Taking 128Gb memory system:
#
# - Without Huge Page:
#    128Gb / 4Kb = 32.7 millions of pages
# => High miss probability
#
# - With Huge Page:
#    128 Gb / 2Mo = 64000 of pages
# => Higher hit probability

  if [ $1 -eq ${BOOL_FALSE} ]; then
    print "Enabling Transparent Huge Page"
    HUGE_PAGES_VALUE="never"
  else
    print "Disabling Transparent Huge Page"
    HUGE_PAGES_VALUE="always"
  fi

  # Debian path
  if [ -f ${HUGE_PAGES_DEBIAN_PATH} ]; then
    HUGES_PAGES_PATH=${HUGE_PAGES_DEBIAN_PATH}
    echo ${HUGE_PAGES_VALUE} > "${HUGE_PAGES_DEBIAN_PATH}"

  # RedHat path
  elif [ -f ${HUGE_PAGES_REDHAT_PATH} ]; then
    HUGE_PAGES_PATH=${HUGE_PAGES_REDHAT_PATH}
    echo ${HUGE_PAGES_VALUE} > "${HUGE_PAGES_REDHAT_PATH}"

  # No match - let's do nothing
  else
    print "Unable to find Transparent Huge Page at the following paths:"
    print "\t-${HUGE_PAGES_DEBIAN_PATH}"
    print "\t-${HUGE_PAGES_REDHAT_PATH}"
    return
  fi

  print "Transparent Huge Page status: cat ${HUGE_PAGES_PATH}"
}

# Disable Huge Page by default
set_hugepage_status $BOOL_FALSE

# Parse options
while getopts vt:o:h-: OPT; do
    # support long options: https://stackoverflow.com/a/28466267/519360
    if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
      OPT="${OPTARG%%=*}"       # extract long option name
      OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
      OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=
    fi

    case $OPT in
      v | verbose)
        VERBOSE=$BOOL_TRUE
        print "Verbose mode enabled"
        ;;
      t | tcmalloc)
        shift;
        TCMALLOC_LIBRARY=$OPTARG
        if [ ! -f $TCMALLOC_LIBRARY ]; then
            echo "tcmalloc library path ${TCMALLOC_LIBRARY} doesn't exist"
            exit 4
        fi
        print "Adding $TCMALLOC_LIBRARY to LD_PRELOAD"
        LD_PRELOAD=$TCMALLOC_LIBRARY:$LD_PRELOAD
        ;;
      o | openmp)
        shift
        OMP_LIBRARY=$OPTARG
        if [ ! -f $OMP_LIBRARY ]; then
            echo "OpenMP library path ${OMP_LIBRARY} doesn't exist"
            exit 3
        fi
        print "Adding $OMP_LIBRARY to LD_PRELOAD"
        LD_PRELOAD=$OMP_LIBRARY:$LD_PRELOAD
        ;;
      hugepage)
        set_hugepage_status $BOOL_TRUE
        ;;
      ??* )
        echo "Illegal option --$OPT"
        exit 2
        ;;
      ? )
        exit 1
        ;;
      -- )
        shift
        break
        ;;
    esac
done

# Remove parsed options
shift $((OPTIND-1))

echo "Python arguments: $@"

# Run the benchmark
export LD_PRELOAD
exec python src/main.py $@